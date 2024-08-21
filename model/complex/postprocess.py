import numpy as np
import os
import torch
from tqdm import tqdm
import argparse
from preprocessing import sg_graph_dict
import random



parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to dataset')
parser.add_argument('--thresh', type=float, help='Threshold for link prediction')
parser.add_argument('--sg_loc', type=str, help='Location of scene graph data')
parser.add_argument('--dataset', type=str, help='dataset name')

args = parser.parse_args()
path_dataset = args.path
THRESHOLD = args.thresh 


def assign_train_set(sg_location, dataset_name, split_ratio, seed, LOC):
    entity_mmtriples = sg_graph_dict(sg_location, dataset_name, LOC)
    generator = torch.Generator().manual_seed(seed)
    if dataset_name == "fb15k":
        a = random.sample(list(entity_mmtriples.keys()),747)
    elif dataset_name == "fb15k-237":
        a = random.sample(list(entity_mmtriples.keys()),725)
    elif dataset_name == "YAGO15K":
        a = random.sample(list(entity_mmtriples.keys()),764)
    elif dataset_name == "DB15K":
        a = random.sample(list(entity_mmtriples.keys()),738)
    mmtriples_train, mmtriples_valid, mmtriples_test = a,a,a
    return mmtriples_train, mmtriples_valid, mmtriples_test, entity_mmtriples

mmtriples_train, mmtriples_valid, mmtriples_test, entity_mmtriples = assign_train_set(args.sg_loc, args.dataset, [0.6,0.1,0.3], 10, args.path)

def file_generator(mode, mmtriples_mode):
    
    os.system(f'cp {path_dataset}/{mode}.txt {path_dataset}/{mode}_withimg2.txt')
    with open(f'{path_dataset}/{mode}_withimg2.txt','a') as fw:
        for key in tqdm(mmtriples_mode):
            for value in entity_mmtriples[key]:
                fw.write(f'{key}_{value[0]}\t{value[1]}\t{key}_{value[2]}\n')

    os.system(f'cp {path_dataset}/{mode}_withimg2.txt {path_dataset}/{mode}_connected2.txt')
file_generator('train', mmtriples_train)
file_generator('valid', mmtriples_valid)
file_generator('test', mmtriples_test)


data_files = os.listdir(f'{path_dataset}/test_doc/')


ftrain = open(f'{path_dataset}/train_connected2.txt','a')
fvalid = open(f'{path_dataset}/valid_connected2.txt','a')
ftest = open(f'{path_dataset}/test_connected2.txt','a')

if args.dataset=='fb15k' or args.dataset=='fb15k-237':
    mmtriples_train = [x.replace('/','.')[1:] for x in mmtriples_train]
    mmtriples_valid = [x.replace('/','.')[1:] for x in mmtriples_valid]
    mmtriples_test = [x.replace('/','.')[1:] for x in mmtriples_test]
for i in tqdm(mmtriples_train):
    data_file = torch.load(f'{path_dataset}/test_doc/{i}.pt')
    link_prob = torch.load(f'{path_dataset}/result_doc/results_{i}.pt')['preds']
    data_file_s = [data_file[s] for s in np.where(link_prob>=THRESHOLD)[0]]
    counter=0
    for triplet in data_file_s:
        counter+=1
        if counter==100:
            break
        ftrain.write(f'{triplet[0]}\t{triplet[1]}\t{triplet[2]}\n')
        fvalid.write(f'{triplet[0]}\t{triplet[1]}\t{triplet[2]}\n')
        ftest.write(f'{triplet[0]}\t{triplet[1]}\t{triplet[2]}\n')
    
    
ftrain.close()
fvalid.close()
ftest.close()

np.save(f'{path_dataset}/meta2.npy',{'entity_mmtriples':entity_mmtriples, 'mmtriples_train': mmtriples_train, 'mmtriples_valid': mmtriples_valid, 'mmtriples_test': mmtriples_test})




