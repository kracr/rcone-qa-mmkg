#!/bin/sh

python postprocess.py --path '../../dataset/FB15k-multimodal/MMKG_th' --thresh 1.0 --sg_loc '../results/48/datasets/' --dataset 'fb15k'


#python postprocess.py --path '../../dataset/FB15k-237-multimodal/' --thresh 1.0 --sg_loc '../results/48/datasets/' --dataset 'fb15k-237'

#python postprocess.py --path '../../dataset/YAGO15K/' --thresh 0.01 --sg_loc '../results/48/datasets/' --dataset 'YAGO15K'

#python postprocess.py --path '../../dataset/DB15K/' --thresh 0.99 --sg_loc '../results/48/datasets/' --dataset 'DB15K'

