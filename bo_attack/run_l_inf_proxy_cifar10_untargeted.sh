#!/bin/bash
GPU=$1
CUDA_VISIBLE_DEVICES=${GPU} python l_inf_proxy_attack.py --dset cifar10 \
    --arch resnet50 --iter 1000 --eps 0.05 --dim 12 --num_attacks 1000 \
    --channel 3 --hard_label --optimize_acq scipy --cos --sin
