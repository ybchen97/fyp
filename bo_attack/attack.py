# pylint: disable=E1101, E0401, E1102
import time
import random
import argparse
import pickle
import numpy as np
import torch
import sys
import warnings

from models import MNIST, CIFAR10, IMAGENET, load_model
from utils import load_mnist_data, load_cifar10_data, load_imagenet_data
from utils import proj, latent_proj, transform

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qExpectedImprovement, ExpectedImprovement, PosteriorMean
from botorch.acquisition import ProbabilityOfImprovement, UpperConfidenceBound
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim import joint_optimize, gen_batch_initial_conditions
from botorch.gen import gen_candidates_torch, get_best_candidates


def obj_func(x, x0, y0):
    """
    Evaluate the objective function.
    if hard label:
        -1 if image is correctly classified,
        0 otherwise
        (done this way because BayesOpt assumes we want to maximize)
    if soft label:
        correct logit - highest logit other than correct logit
    in both cases, successful adversarial perturbation iff
    objective function >= 0

    Args:
        x: [n, latent_dim] size float tensor. Represents the perturbations.
        x0: Image to attack
        y0: Label of image to attack

    Returns:
        Objective function value
    """
    # Transforms low dimensional perturbation into actual perturbation through
    # irfft
    x = transform(x, args.dset, args.arch, args.cos, args.sin).to(device)
    x = proj(x, args.eps, args.inf_norm, args.discrete)
    with torch.no_grad():
        y = cnn_model.predict_scores(x + x0)

    if not args.hard_label:
        y = torch.log_softmax(y, dim=1)
        max_score = y[:, y0]
        y, index = torch.sort(y, dim=1, descending=True)
        select_index = (index[:, 0] == y0).long()
        next_max = y.gather(1, select_index.view(-1, 1)).squeeze()
        f = torch.max(max_score - next_max, torch.zeros_like(max_score))
    else:
        index = torch.argmax(y, dim=1)
        f = torch.where(index == y0, torch.ones_like(index),
                        torch.zeros_like(index)).float()
    return -f


def initialize_model(x0, y0, n=5):
    """
    Initialize botorch GP model.

    Args:
        x0: image to attack
        y0: label of image to attack
        n (int): initial number of perturbations drawn to form the prior for the
            GP

    Returns:
        train_x: [n, latent_dim] size float tensor. Represents the perturbations
        train_obj:
        mll:
        model:
        best_observed_value:
        mean:
        std:
    """

    # generate prior xs and ys for GP
    # Draw n perturbations to form prior for the GP. train_x is the
    # perturbations
    train_x = 2 * torch.rand(n, latent_dim, device=device).float() - 1  # drawing from uniform range [-1,1]
    if not args.inf_norm:
        train_x = latent_proj(train_x, args.eps)
    train_obj = obj_func(train_x, x0, y0)
    mean, std = train_obj.mean(), train_obj.std()
    if args.standardize:  # not used
        train_obj = (train_obj - train_obj.mean()) / train_obj.std()
    best_observed_value = train_obj.max().item()

    # define models for objective and constraint
    model = SingleTaskGP(train_X=train_x, train_Y=train_obj[:, None])
    model = model.to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll = mll.to(train_x)
    return train_x, train_obj, mll, model, best_observed_value, mean, std


def optimize_acqf_and_get_observation(acq_func, x0, y0):
    # Optimizes the acquisition function, returns new candidate new_x
    # and its objective function value new_obj

    # optimize
    if args.optimize_acq == 'scipy':
        candidates, acq_value = joint_optimize(
            acq_function=acq_func,
            bounds=bounds,
            q=args.q,
            num_restarts=args.num_restarts,
            raw_samples=200,
        )
    else:
        Xinit = gen_batch_initial_conditions(
            acq_func,
            bounds,
            q=args.q,
            num_restarts=args.num_restarts,
            raw_samples=500
        )
        batch_candidates, batch_acq_values = gen_candidates_torch(
            initial_conditions=Xinit,
            acquisition_function=acq_func,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
            verbose=False
        )
        candidates = get_best_candidates(batch_candidates, batch_acq_values)

    # observe new values
    new_x = candidates.detach()
    if not args.inf_norm:
        new_x = latent_proj(new_x, args.eps)
    new_obj = obj_func(new_x, x0, y0)
    return new_x, new_obj


def bayes_opt(x0, y0):
    """
    Main Bayesian optimization loop. Begins by initializing model, then for each
    iteration, it fits the GP to the data, gets a new point with the acquisition
    function, adds it to the dataset, and exits if it's a successful attack

    Args:
        x0: image to attack
        y0: true label of image x0 to attack

    Returns:
        A tuple of (query_count (int), success (Boolean))

    """

    best_observed = []
    query_count, success = 0, False

    # call helper function to initialize model
    train_x, train_obj, mll, model, best_value, mean, std = initialize_model(
        x0, y0, n=args.initial_samples)
    if args.standardize_every_iter:  # not used
        train_obj = (train_obj - train_obj.mean()) / train_obj.std()
    best_observed.append(best_value)
    query_count += args.initial_samples

    # run args.iter rounds of BayesOpt after the initial random batch
    for _ in range(args.iter):

        # fit the model
        fit_gpytorch_model(mll)

        # define the qNEI acquisition module using a QMC sampler
        if args.q != 1:
            qmc_sampler = SobolQMCNormalSampler(num_samples=2000,
                                                seed=seed)
            qEI = qExpectedImprovement(model=model, sampler=qmc_sampler,
                                       best_f=best_value)
        else:
            if args.acqf == 'EI':
                qEI = ExpectedImprovement(model=model, best_f=best_value)
            elif args.acqf == 'PM':
                qEI = PosteriorMean(model)
            elif args.acqf == 'POI':
                qEI = ProbabilityOfImprovement(model, best_f=best_value)
            elif args.acqf == 'UCB':
                qEI = UpperConfidenceBound(model, beta=args.beta)

        # optimize and get new observation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_x, new_obj = optimize_acqf_and_get_observation(qEI, x0, y0)

        if args.standardize:  # not used
            new_obj = (new_obj - mean) / std

        # update training points
        train_x = torch.cat((train_x, new_x))
        train_obj = torch.cat((train_obj, new_obj))
        if args.standardize_every_iter:  # not used
            train_obj = (train_obj - train_obj.mean()) / train_obj.std()

        # update progress
        best_value, best_index = train_obj.max(0)
        best_observed.append(best_value.item())
        best_candidate = train_x[best_index]

        # reinitialize the model so it is ready for fitting on next iteration
        torch.cuda.empty_cache()
        model.set_train_data(train_x, train_obj, strict=False)

        # get objective value of best candidate; if we found an adversary, exit
        best_candidate = best_candidate.view(1, -1)
        best_candidate = transform(
            best_candidate, args.dset, args.arch, args.cos, args.sin).to(device)
        best_candidate = proj(best_candidate, args.eps,
                              args.inf_norm, args.discrete)
        with torch.no_grad():
            adv_label = torch.argmax(
                cnn_model.predict_scores(best_candidate + x0))
        if adv_label != y0:
            success = True
            if args.inf_norm:
                print('Adversarial Label', adv_label.item(),
                      'Norm:', best_candidate.abs().max().item())
            else:
                print('Adversarial Label', adv_label.item(),
                      'Norm:', best_candidate.norm().item())
            return query_count, success
        query_count += args.q
    # not successful (ran out of query budget)
    return query_count, success


def attack_bayes():
    """
    Perform BayesOpt attack
    """

    # get dataset and list of indices of images to attack
    if args.dset == 'mnist':
        test_dataset = load_mnist_data()
        samples = np.arange(args.start, args.start + args.num_attacks)

    elif args.dset == 'cifar10':
        test_dataset = load_cifar10_data()
        samples = np.arange(args.start, args.start + args.num_attacks)

    elif args.dset == 'imagenet':
        if args.arch == 'inception_v3':
            test_dataset = load_imagenet_data(299, 299)
        else:
            test_dataset = load_imagenet_data()

        # see readme
        samples = np.load('random_indices_imagenet.npy')
        samples = samples[args.start: args.start + args.num_attacks]

    else:
        test_dataset = np.array([])
        samples = np.array([])
        sys.exit(f"{args.dset} is not available!")

    print("Length of sample_set: ", len(samples))
    results_dict = {}
    # loop over images, attacking each one if it is initially correctly classified
    for idx in samples[:args.num_attacks]:
        image, label = test_dataset[idx]
        image = image.unsqueeze(0).to(device)
        predicted_label = torch.argmax(cnn_model.predict_scores(image))
        print(f"Image {idx}, Label: {label}, Predicted label: ",
              predicted_label.item())

        # ignore incorrectly classified images
        if label == predicted_label:
            # itr, success = bayes_opt(image, label)
            retry = 0
            while retry < 5:
                try:
                    query_count, success = bayes_opt(image, label)
                    break
                except Exception as e:
                    print(e)
                    print('Retry_{}'.format(retry))
                    retry += 1
            if success:
                print(f"Succeeded, queries: {query_count}")
                results_dict[idx] = query_count
            else:
                print(f"Failed, queries: {query_count}")
                results_dict[idx] = 0

        sys.stdout.flush()

    # results saved as dictionary, with entries of the form
    # dataset idx : 0 if unsuccessfully attacked, # of queries if successfully attacked
    print('RESULTS', results_dict)
    if args.save:
        pickle.dump(results_dict,
                    open(f"{args.dset:s}{args.arch:s}_{args.start:d}_{args.iter:d}_{args.dim:d}_{args.eps:.2f}_{args.num_attacks:d}.pkl", 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # network architecture (resnet50, vgg16_bn, or inception_v3)
    parser.add_argument('--arch', type=str, default='resnet50')
    # BayesOpt acquisition function
    parser.add_argument('--acqf', type=str, default='EI')
    # hyperparam for UCB acquisition function
    parser.add_argument('--beta', type=float, default=1.0)
    # number of channels in image
    parser.add_argument('--channel', type=int, default=1)
    parser.add_argument('--dim', type=int, default=784)  # dimension of attack
    # dataset to attack
    parser.add_argument('--dset', type=str, default='cifar10')
    # if True, project to boundary of epsilon ball (instead of just projecting
    # inside)
    parser.add_argument('--discrete', default=False, action='store_true')
    # bound on perturbation norm
    parser.add_argument('--eps', type=float, default=0.3)
    # hard-label vs soft-label attack
    parser.add_argument('--hard_label', default=False, action='store_true')
    # number of BayesOpt iterations to perform
    parser.add_argument('--iter', type=int, default=1)
    # number of samples taken to form the GP prior
    parser.add_argument('--initial_samples', type=int, default=5)
    parser.add_argument('--inf_norm', default=False,
                        action='store_true')  # perform L_inf norm attack
    # number of images to attack
    parser.add_argument('--num_attacks', type=int, default=1)
    # hyperparam for acquisition function
    parser.add_argument('--num_restarts', type=int, default=1)
    # backend for acquisition function optimization (torch or scipy)
    parser.add_argument('--optimize_acq', type=str, default='torch')
    # number of candidates to receive from acquisition function
    parser.add_argument('--q', type=int, default=1)
    # index of first image to attack
    parser.add_argument('--start', type=int, default=0)
    # save dictionary of results at end
    parser.add_argument('--save', default=False, action='store_true')
    parser.add_argument('--standardize', default=False,
                        action='store_true')  # normalize objective values
    # normalize objective values at every BayesOpt iteration
    parser.add_argument('--standardize_every_iter',
                        default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=1)  # random seed
    # if True, use sine FFT basis vectors
    parser.add_argument('--sin', default=False, action='store_true')
    # if True, use cosine FFT basis vectors
    parser.add_argument('--cos', default=False, action='store_true')
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    if args.dset == 'mnist':
        net = MNIST()
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        load_model(net, 'models/mnist_gpu.pt')
        net.eval()
        cnn_model = net.module
    elif args.dset == 'cifar10':
        net = CIFAR10()
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        load_model(net, 'models/cifar10_gpu.pt')
        net.eval()
        cnn_model = net.module
    elif args.dset == 'imagenet':
        cnn_model = IMAGENET(args.arch)

    timestart = time.time()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if args.sin and args.cos:
        latent_dim = args.dim * args.dim * args.channel * 2
    else:
        latent_dim = args.dim * args.dim * args.channel

    bounds = torch.tensor([[-2.0] * latent_dim, [2.0] * latent_dim],
                          device=device).float()
    attack_bayes()
    timeend = time.time()
    print("\n\nTotal running time: %.4f seconds\n" % (timeend - timestart))
