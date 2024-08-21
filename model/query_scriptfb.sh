#!/bin/usr/env bash

python create_queries.py --index_only --gen_train_num 273710 --gen_test_num 8000 \
 --gen_valid_num 8000 --gen_train --gen_test --gen_valid --gen_id 0 \
 --dataset 'fb15k' --sg_loc 'results/48/datasets' \
 --path '../dataset/FB15k-multimodal/MMKG_th/' \
 --query_folder 'query_generation' --max_ans_num 100 --per_mm 0.3

python create_queries.py --gen_train_num 273710 --gen_test_num 8000 \
 --gen_valid_num 8000 --gen_train --gen_test --gen_valid --gen_id 0 \
 --dataset 'fb15k' --sg_loc 'results/48/datasets' \
 --path '../dataset/FB15k-multimodal/MMKG_th/' \
 --query_folder 'query_generation' --max_ans_num 100 --per_mm 0.3

for i in 1 2 3 4
do
  python create_queries.py --gen_train_num 289710 \
  --gen_train --gen_id $i \
  --dataset 'fb15k' --sg_loc 'results/48/datasets' \
  --path '../dataset/FB15k-multimodal/MMKG_th/' \
  --query_folder 'query_generation' --max_ans_num 100 --per_mm 0.3

  python create_queries.py --gen_test_num 5600 \
  --gen_test --gen_valid --gen_valid_num 5600 --gen_id $i \
  --dataset 'fb15k' --sg_loc 'results/48/datasets' \
  --path '../dataset/FB15k-multimodal/MMKG_th/' \
  --query_folder 'query_generation' --max_ans_num 100 --per_mm 0.0
done

for i in 5 6
do 
  python create_queries.py --gen_train_num 2400 \
  --gen_train --gen_id $i \
  --dataset 'fb15k' --sg_loc 'results/48/datasets' \
  --path '../dataset/FB15k-multimodal/MMKG_th/' \
  --query_folder 'query_generation' --max_ans_num 100 --per_mm 1.0

  python create_queries.py --gen_test_num 5600 \
  --gen_test --gen_id $i \
  --dataset 'fb15k' --sg_loc 'results/48/datasets' \
  --path '../dataset/FB15k-multimodal/MMKG_th/' \
  --query_folder 'query_generation' --max_ans_num 100 --per_mm 0.0
done

for i in 10 11
do
  python create_queries.py --gen_valid_num 5600 \
  --gen_valid --gen_test_num 5600 --gen_test --gen_id $i \
  --dataset 'fb15k' --sg_loc 'results/48/datasets' \
  --path '../dataset/FB15k-multimodal/MMKG_th/' \
  --query_folder 'query_generation' --max_ans_num 100 --per_mm 0.0
  
  python create_queries.py --gen_train_num 43371 \
  --gen_train --gen_id $i \
  --dataset 'fb15k' --sg_loc 'results/48/datasets' \
  --path '../dataset/FB15k-multimodal/MMKG_th/' \
  --query_folder 'query_generation' --max_ans_num 100 --per_mm 0.3
done

for i in 7 8 9
do
  python create_queries.py --gen_train_num 43371 \
  --gen_train --gen_id $i \
  --dataset 'fb15k' --sg_loc 'results/48/datasets' \
  --path '../dataset/FB15k-multimodal/MMKG_th/' \
  --query_folder 'query_generation' --max_ans_num 100 --per_mm 0.3
  

  python create_queries.py --gen_test_num 5600 \
  --gen_test --gen_valid --gen_valid_num 5600 --gen_id $i \
  --dataset 'fb15k' --sg_loc 'results/48/datasets' \
  --path '../dataset/FB15k-multimodal/MMKG_th/' \
  --query_folder 'query_generation' --max_ans_num 100 --per_mm 0.0
done


for i in 12 13
do
  
  python create_queries.py --gen_test_num 5600 \
  --gen_test --gen_id $i \
  --dataset 'fb15k' --sg_loc 'results/48/datasets' \
  --path '../dataset/FB15k-multimodal/MMKG_th/' \
  --query_folder 'query_generation' --max_ans_num 100 --per_mm 0.0

  python create_queries.py --gen_train_num 2400 \
  --gen_train --gen_id $i \
  --dataset 'fb15k' --sg_loc 'results/48/datasets' \
  --path '../dataset/FB15k-multimodal/MMKG_th/' \
  --query_folder 'query_generation' --max_ans_num 100 --per_mm 1.0
done


python create_queries.py --post_combine --gen_train_num 273710 --gen_test_num 8000 \
 --gen_valid_num 8000 --gen_train --gen_test --gen_valid --gen_id 0 \
 --dataset 'fb15k' --sg_loc 'results/48/datasets' \
 --path '../dataset/FB15k-multimodal/MMKG_th/' \
 --query_folder 'query_generation' --max_ans_num 100 --per_mm 0.3
