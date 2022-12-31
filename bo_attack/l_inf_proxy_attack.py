# pylint: disable=E1101, E0401, E1102
import time
import random
import argparse
import pickle
import numpy as np
import torch
import sys
import os
import warnings

from models import MNIST, CIFAR10, IMAGENET, load_model
from utils import load_mnist_data, load_cifar10_data, load_imagenet_data
from utils import proj_inf, latent_proj, transform

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qExpectedImprovement, ExpectedImprovement, PosteriorMean
from botorch.acquisition import ProbabilityOfImprovement, UpperConfidenceBound
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim import joint_optimize, gen_batch_initial_conditions
from botorch.gen import gen_candidates_torch, get_best_candidates


def fine_grained_binary_search_local(model, x0, y0, theta, initial_lamb=1.0,
    tol=1e-5):
    nquery = 0
    lbd = initial_lamb

    if model.predict_label(x0+lbd*theta) == y0:
        lbd_lo = lbd
        lbd_hi = lbd*1.01
        nquery += 1
        while model.predict_label(x0+lbd_hi*theta) == y0:
            lbd_hi = lbd_hi*1.01
            nquery += 1
            if lbd_hi > 20:
                print("Bad initialization direction.")
                return float('inf'), theta, nquery
    else:
        lbd_hi = lbd
        lbd_lo = lbd*0.99
        nquery += 1
        while model.predict_label(x0+lbd_lo*theta) != y0 :
            lbd_lo = lbd_lo*0.99
            nquery += 1

    while (lbd_hi - lbd_lo) > tol:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if model.predict_label(x0 + lbd_mid*theta) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, lbd_hi*theta, nquery


def obj_func_proxy(x, x0, y0):
    """
    Evaluate the objective function.
    Objective function returns the distance to decision boundary along search
    direction. Take negative since BayesOpt wants to maximize

    Args:
        x: [1, latent_dim] size float tensor. Represents the perturbations
        x0: Image to attack
        y0: Label of image to attack

    Returns:
        -d: Distance to decision boundary along direction theta
        n_query: Number of queries taken
    """
    # Low dimensional perturbation will be transformed into actual perturbation
    # to be added to the image
    x = transform(x, args.dset, args.arch, args.cos, args.sin).to(device)
    x = proj_inf(x, args.eps, args.discrete)
    #x /= x.abs().max().item()

    with torch.no_grad():
        d, perturb, n_query = fine_grained_binary_search_local(cnn_model, x0, y0, x)

    if args.verbose:
        print(f"Distance: {d}, norm: {perturb.abs().max().item()}, {n_query} queries taken.")

    return torch.tensor([-d]).to(device), perturb, n_query


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
        train_perturb:
        mll:
        model:
        best_observed_value:
        n_query:
    """

    # generate prior xs and ys for GP
    # Draw n perturbations to form prior for the GP. train_x is the
    # perturbations
    train_x = 2 * torch.rand(n, latent_dim, device=device).float() - 1  # drawing from uniform range [-1,1]

    n_query = 0
    train_obj = []
    train_perturb = []
    for i in range(n):
        obj, perturb, n_q = obj_func_proxy(train_x[i:i+1], x0, y0)
        train_obj.append(obj)
        train_perturb.append(perturb)
        n_query += n_q
    train_obj = torch.cat(train_obj)
    train_perturb = torch.cat(train_perturb)

    best_observed_value = train_obj.max().item()

    # define models for objective and constraint
    model = SingleTaskGP(train_X=train_x, train_Y=train_obj[:, None])
    model = model.to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll = mll.to(train_x)
    return train_x, train_perturb, train_obj, mll, model, best_observed_value, n_query


def optimize_acqf_and_get_observation(acq_func, x0, y0):
    """
    Optimizes the acquisition function, returns new candidate new_x and its
    objective function value new_obj

    Args:
        acq_func: Chosen acquisition function
        x0: Image to attack
        y0: Label of image to attack
    
    Returns:
        new_x:
        new_obj:
        n_query:
    """

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

    # observe new values, size is [1, latent_dim]
    new_x = candidates.detach()
    new_obj, new_perturb, n_query = obj_func_proxy(new_x, x0, y0)
    return new_x, new_perturb, new_obj, n_query


def bayes_opt(x0, y0, tol=1e-5):
    """
    Main Bayesian optimization loop. Begins by initializing model, then for each
    iteration, it fits the GP to the data, gets a new point with the acquisition
    function, adds it to the dataset, and exits if the norm of the perturbation
    is less than the epsilon value.

    Args:
        x0: Image to attack
        y0: True label of image x0 to attack

    Returns:
        A tuple of (query_count (int), success (Boolean))
    """

    best_observed = []
    best_norm = 0.0
    query_count = 0
    n_sample = args.initial_samples
    img_results = {
        'img': x0.cpu().detach().numpy(),
        'label': y0,
        'adv_label': -1,
        'adv_perturbation': np.zeros(x0.size())
    }

    # call helper function to initialize model
    train_x, train_perturb, train_obj, mll, model, best_value, n_query = initialize_model(
        x0, y0, n=args.initial_samples)
    best_observed.append(best_value)
    query_count += n_query
    if args.verbose:
        print(f"{n_query} queries taken for initialization with {args.initial_samples} samples.")

    # run args.iter rounds of BayesOpt after the initial random batch
    while query_count < args.iter:

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
            new_x, new_perturb, new_obj, n_query = optimize_acqf_and_get_observation(qEI, x0, y0)
            n_sample += 1
            query_count += n_query

        # update training points
        train_x = torch.cat((train_x, new_x))
        train_perturb = torch.cat((train_perturb, new_perturb))
        train_obj = torch.cat((train_obj, new_obj))

        # update progress
        best_value, best_index = train_obj.max(0)
        best_observed.append(best_value.item())
        best_candidate = train_perturb[best_index]
        best_norm = best_candidate.abs().max().item()

        # reinitialize the model so it is ready for fitting on next iteration
        torch.cuda.empty_cache()
        model.set_train_data(train_x, train_obj, strict=False)

        with torch.no_grad():
            adv_label = cnn_model.predict_label(best_candidate + x0)

        if adv_label == y0:
            print("Something went wrong!")

        if best_norm - args.eps < tol:
            print('Adversarial Label', adv_label.item(), 'Norm:', best_norm)
            img_results['adv_label'] = adv_label.item()
            img_results['adv_perturbation'] = best_candidate.cpu().detach().numpy()
            return query_count, best_norm, n_sample, img_results, True

    # not successful (ran out of query budget)
    return query_count, best_norm, n_sample, img_results, False


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
    incorrect_cls = 0
    # loop over images, attacking each one if it is initially correctly classified
    for idx in samples[:args.num_attacks]:
        image, label = test_dataset[idx]
        image = image.unsqueeze(0).to(device)
        predicted_label = torch.argmax(cnn_model.predict_scores(image))
        print(f"Image {idx}, Label: {label}, Predicted label: ",
              predicted_label.item())

        # ignore incorrectly classified images
        if label == predicted_label:
            query_count, norm, n_sample, img_results, success = bayes_opt(image, label)
            if success:
                print(f"Succeeded, queries: {query_count}, n_sample: {n_sample}, avg queries per sample: {query_count / n_sample}")
            else:
                print(f"Failed, queries: {query_count}")

            results_dict[idx] = { 
                "success": success,
                "queries": query_count, 
                "norm": norm,
                "n_sample": n_sample,
                "avg_queries_per_sample": query_count / n_sample,
                "image_results": img_results
            }
        else:
            incorrect_cls += 1

        sys.stdout.flush()

    # results saved as dictionary, with entries of the form
    # dataset idx : 0 if unsuccessfully attacked, # of queries if successfully attacked
    print(f"{args.num_attacks - incorrect_cls} images chosen for attack. Other {incorrect_cls} images are classified wrongly by model.")

    os.makedirs(args.save_path, exist_ok=True)
    filename = f"{args.dset:s}{args.arch:s}_{args.start:d}_{args.iter:d}_{args.dim:d}_{args.eps:.2f}_{args.num_attacks:d}.pkl"
    pickle.dump(results_dict, open(os.path.join(args.save_path, filename), 'wb'))


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
    # prints distance and number of queries per sample of objective function 
    parser.add_argument('--verbose', default=False, action='store_true')
    # number of BayesOpt iterations to perform
    parser.add_argument('--iter', type=int, default=1)
    # number of samples taken to form the GP prior
    parser.add_argument('--initial_samples', type=int, default=5)
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
    parser.add_argument('--save_path', type=str, default='results')
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
