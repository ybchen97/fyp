# Black-box Hard-Label Adversarial Attacks with Bayesian Optimization

Improving bayesian optimization black-box hard-label attack by adding a distance
to decision boundary proxy. This codebase is based off the [original bayesian
optimization hard-label attack
codebase](https://github.com/satyanshukla/bayes_attack).

To run attack, simply run the scripts e.g. run l_infinity proxy attack on
cifar10 dataset:
```bash
./run_l_inf_proxy_cifar10_untargeted.sh
```

Proxy attack code is written in the following files:
- l2_proxy_attack.py
- l_inf_proxy_attack.py

Prereqs:

* pytorch 1.7.0
* torchvision 0.8.1 (or a version that fits with pytorch 1.7.0)
* botorch 0.2.0 (see https://botorch.org/#quickstart for installation)
* gpytorch 1.4.2

## Credits
1. [Simple and Efficient Hard Label Black-box Adversarial Attacks
in Low Query Budget Regimes](https://github.com/satyanshukla/bayes_attack)
2. [Sign-Opt: A Query-Efficient Hard-Label Adversarial Attack](https://github.com/cmhcbb/attackbox)
