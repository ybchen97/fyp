#!/bin/bash
GPU=$1
CUDA_VISIBLE_DEVICES=${GPU} python attack.py --dset mnist --arch resnet50 \
    --iter 995 --inf_norm --eps 0.3 --dim 12 --num_attacks 1000 --channel 1 \
    --hard_label --optimize_acq scipy --cos --sin --save
