import numpy as np
import networkx as nx
import argparse
from tqdm import tqdm
import os
import torch


def sg_graph_dict(sg_location, dataset_name, LOC):
    idx_to_name = dict(np.load(f'{sg_location}/{dataset_name}/details_all.npy',allow_pickle=True).item())


    entity_mmtriplets = dict()
    entity_list = [d for d in os.listdir(f'{sg_location}/{dataset_name}/') if os.path.isdir(f'{sg_location}/{dataset_name}/{d}')]
    

    if dataset_name == 'YAGO15K' or dataset_name == 'DB15K':
        with open(f'{LOC}/{dataset_name}_Imagedetails.npy','rb') as fr:
            fb15ktodata = np.load(fr, allow_pickle=True).item() 
    
    for entity in tqdm(entity_list):
        if dataset_name == 'fb15k' or dataset_name == 'fb15k-237':
            image = [d for d in os.listdir(f'{sg_location}/{dataset_name}/{entity}/') if not d.startswith('.')][0]
        
            triplet = []
            with open(f'{sg_location}/{dataset_name}/{entity}/{image}/triplets.txt','r') as fr:
                for line in fr.readlines():
                    idx = line.split()
                    triplet.append([idx_to_name['entity_class'][int(idx[0])], idx_to_name['predicate_class'][int(idx[1])], idx_to_name['entity_class'][int(idx[2])]])
            entity_mmtriplets["/"+entity.replace(".","/")]=np.array(triplet)
        elif dataset_name == 'YAGO15K' or dataset_name == 'DB15K':
            image = [d for d in os.listdir(f'{sg_location}/{dataset_name}/{entity}/') if not d.startswith('.')][0]
        
            triplet = []
            
            with open(f'{sg_location}/{dataset_name}/{entity}/{image}/triplets.txt','r') as fr:
                for line in fr.readlines():
                    idx = line.split()
                    triplet.append([idx_to_name['entity_class'][int(idx[0])], idx_to_name['predicate_class'][int(idx[1])], idx_to_name['entity_class'][int(idx[2])]])
            entity_mmtriplets[fb15ktodata[entity]]=np.array(triplet)
        
    return entity_mmtriplets



def file_before_connect(entity_mmtriplets, NEIGHBOUR_HOP, dataset_name, LOC):
    entity_set = []
    relation_set = []
    triplets = []

    with open(f'{LOC}/train.txt','r') as fr:
        for line in fr.readlines():
            x = line.split()
            entity_set.append(x[0])
            entity_set.append(x[2].strip())
            relation_set.append(x[1])
            triplets.append(x)


    entity_set = list(set(entity_set))
    relation_set = list(set(relation_set))



    G = nx.Graph()
    G.add_nodes_from(entity_set)
    for row in triplets:
        G.add_edge(row[0],row[2],relation=row[1])


    os.system(f'cp {LOC}/train.txt {LOC}/train_withimg.txt')
    #i=0

    if not os.path.exists(f'{LOC}/test_doc/'):
        os.makedirs(f'{LOC}/test_doc/')

    with open(f'{LOC}/train_withimg.txt','a') as fw_train_wi:
    
        for key, value in tqdm(entity_mmtriplets.items()):             #change this value accordingly
            neigh = nx.single_source_shortest_path_length(G, key, cutoff=NEIGHBOUR_HOP)
            neigh = list(neigh) 
            entity_list_image = list(np.unique(np.concatenate((value[:,0],value[:,2]))))

            
            if dataset_name == 'fb15k' or dataset_name == 'fb15k-237':
                ent_name = key.replace('/','.')[1:]
            else:
                ent_name = key

            cand_triplet=[]
            for n in neigh:
                for ent in entity_list_image:
                    for rel in relation_set:            
                        cand_triplet.append([str(n), str(rel), str(key)+'_'+str(ent)])
            torch.save(cand_triplet,f'{LOC}/test_doc/{ent_name}.pt')

            for j in range(len(value)):
                fw_train_wi.write(f'{key}_{value[j][0]}\t{value[j][1]}\t{key}_{value[j][2]}\n')


if __name__== "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--loc', type=str, help='Location of original dataset')
    parser.add_argument('--sg_loc', type=str, help='Location of scene graph data')
    parser.add_argument('--hop', type=int, help='number of hops to consider for connections')
    parser.add_argument('--dataset', type=str, help='dataset name')

    args = parser.parse_args()
    
    LOC = args.loc
    sg_location = args.sg_loc
    NEIGHBOUR_HOP = args.hop
    dataset_name = args.dataset
    

    entity_mmtriplets = sg_graph_dict(sg_location, dataset_name, LOC)
    
    file_before_connect(entity_mmtriplets, NEIGHBOUR_HOP, dataset_name, LOC)




