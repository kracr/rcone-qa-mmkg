#!/bin/sh

#python preprocessing.py --loc '../../dataset/FB15k-multimodal/MMKG_th/' --sg_loc '../results/48/datasets' --hop 1 --dataset 'fb15k'

#python preprocessing.py --loc '../../dataset/YAGO15K/' --sg_loc '../results/48/datasets' --hop 1 --dataset 'YAGO15K'

#python preprocessing.py --loc '../../dataset/FB15k-237-multimodal/' --sg_loc '../results/48/datasets' --hop 1 --dataset 'fb15k-237'

python preprocessing.py --loc '../../dataset/DB15K/' --sg_loc '../results/48/datasets' --hop 1 --dataset 'DB15K'

