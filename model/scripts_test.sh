#CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test --checkpoint \
#--data_path ../dataset/FB15k-multimodal/MMKG_th/query_generation  -n 128 -b 512 -d 800 -g 30 --data fb15k \
#-lr 0.00005 --max_steps 450001 --cpu_num 5 --valid_steps 60000 --test_batch_size 10 --save_checkpoint_steps 1000 \
#--seed 0 --drop 0.05 --tag final

#CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test --checkpoint \
#--data_path ../dataset/YAGO15K/query_generation  -n 128 -b 512 -d 800 -g 30 --data YAGO15K \
#-lr 0.00005 --max_steps 450001 --cpu_num 5 --valid_steps 60000 --test_batch_size 10 --save_checkpoint_steps 10000 \
#--seed 0 --drop 0.05 --tag ijyago

#CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test --checkpoint \
#--data_path ../dataset/FB15k-237-multimodal/query_generation  -n 128 -b 512 -d 800 -g 30 --data fb15k-237 \
#-lr 0.00005 --max_steps 450001 --cpu_num 5 --valid_steps 60000 --test_batch_size 5 --save_checkpoint_steps 10000 \
#--seed 0 --drop 0.05 --tag ijfb15k237

CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_test --checkpoint \
--data_path ../dataset/DB15K/query_generation  -n 128 -b 512 -d 800 -g 30 --data DB15K \
-lr 0.00005 --max_steps 450001 --cpu_num 5 --valid_steps 60000 --test_batch_size 5 --save_checkpoint_steps 10000 \
--seed 0 --drop 0.05 --tag ijdb15k

