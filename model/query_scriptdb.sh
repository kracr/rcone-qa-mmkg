#!/bin/usr/env bash

python create_queries.py --index_only --gen_train_num 273710 \
 --gen_train --gen_id 0 \
 --dataset 'DB15K' --sg_loc 'results/48/datasets' \
 --path '../dataset/DB15K/' \
 --query_folder 'query_generation' --max_ans_num 100 --per_mm 0.3

python create_queries.py --gen_train_num 123982 \
 --gen_train --gen_id 0 \
 --dataset 'DB15K' --sg_loc 'results/48/datasets' \
 --path '../dataset/DB15K/' \
 --query_folder 'query_generation' --max_ans_num 100 --per_mm 0.3

for i in 1 2 3 4
do
  python create_queries.py --gen_train_num 123982 \
  --gen_train --gen_id $i \
  --dataset 'DB15K' --sg_loc 'results/48/datasets' \
  --path '../dataset/DB15K/' \
  --query_folder 'query_generation' --max_ans_num 100 --per_mm 0.3
done

for i in 5 6
do 
  python create_queries.py --gen_train_num 8000 \
  --gen_train --gen_id $i \
  --dataset 'DB15K' --sg_loc 'results/48/datasets' \
  --path '../dataset/DB15K/' \
  --query_folder 'query_generation' --max_ans_num 100 --per_mm 0.3
done

for i in 10 11
do
  python create_queries.py --gen_train_num 26798 \
  --gen_train --gen_id $i \
  --dataset 'DB15K' --sg_loc 'results/48/datasets' \
  --path '../dataset/DB15K/' \
  --query_folder 'query_generation' --max_ans_num 100 --per_mm 0.3
done

for i in 7 8 9
do
  python create_queries.py --gen_train_num 26798 \
  --gen_train --gen_id $i \
  --dataset 'DB15K' --sg_loc 'results/48/datasets' \
  --path '../dataset/DB15K/' \
  --query_folder 'query_generation' --max_ans_num 100 --per_mm 0.3
done


for i in 12 13
do
  python create_queries.py --gen_train_num 8000 \
  --gen_train --gen_id $i \
  --dataset 'DB15K' --sg_loc 'results/48/datasets' \
  --path '../dataset/DB15K/' \
  --query_folder 'query_generation' --max_ans_num 100 --per_mm 0.3
done


python create_queries.py --post_combine --gen_train_num 273710 --gen_test_num 8000 \
 --gen_valid_num 8000 --gen_train --gen_test --gen_valid --gen_id 0 \
 --dataset 'DB15K' --sg_loc 'results/48/datasets' \
 --path '../dataset/DB15K/' \
 --query_folder 'query_generation' --max_ans_num 100 --per_mm 0.3
