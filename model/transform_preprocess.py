import torch
import numpy as np
import pickle
from tqdm import tqdm
import scipy.io
import os

def dataload():
    
    data_path = '../dataset/DB15K/query_generation/'
    
    data_path2 = '../dataset/DB15K/'
    dataset = 'DB15K'

    meta = np.load(data_path2+'meta2.npy',allow_pickle=True).item()
    
    fr = open(f'{data_path}/ent2id.pkl','rb')
    ent2id = pickle.load(fr)

    words = np.load('glove/vocab_dict.npy', allow_pickle=True).item()
    vocab = torch.load('glove/glove.pt')
    if dataset == "YAGO15K" or dataset == "DB15K":
        with open(f'{data_path2}{dataset}_Imagedetailsr.npy','rb') as fr:
            d2fb = np.load(fr, allow_pickle=True).item()
        _dir = [d for d in os.listdir(f'results/32/datasets/complex_embeddings/{dataset}') if (not (d.startswith('.')) and d.endswith('.mat')) and d[:-4] in [d2fb[vl] for vl in meta['mmtriples_train']]]
    else:
        _dir = [d for d in os.listdir(f'results/32/datasets/complex_embeddings/{dataset}') if (not (d.startswith('.')) and d.endswith('.mat')) and d[:-4] in meta['mmtriples_train']]
    subentities_orig = np.load(f'results/48/datasets/{dataset}/details_all.npy', allow_pickle = True).item()
    subentities_det = np.load(f'results/32/datasets/{dataset}/details_all.npy', allow_pickle = True).item()

    id2id_arr = dict()
    for d in tqdm(_dir):
        id2id = dict()

        mat = scipy.io.loadmat(f'results/32/datasets/complex_embeddings/{dataset}/{d}')
        torch.save(torch.cat((torch.tensor(mat['entities_real']), torch.tensor(mat['entities_imag'])), 1), f'results/32/datasets/complex_embeddings/{dataset}/{d[:-4]}.pt')

        id2idx_se = np.load(f'results/32/datasets/complex_embeddings/{dataset}/{d[:-4]}.npy', allow_pickle=True).item()
        img_fold = os.listdir(f'results/48/datasets/{dataset}/{d[:-4]}')[0]
        id2idx_se_orig = np.load(f'results/48/datasets/{dataset}/{d[:-4]}/{img_fold}/entities.npy', allow_pickle=True).item()
        ent_det_list = list(id2idx_se.keys())
        ent_orig_list = list(id2idx_se_orig)
        word_emb_orig, word_emb_det = None, None
        for key in ent_orig_list:
            if word_emb_orig is None:
                word_emb_orig = torch.unsqueeze(vocab[int(words[subentities_orig['entity_class'][int(key)]])], 0)
            else:
                word_emb_orig = torch.concat((word_emb_orig, torch.unsqueeze(vocab[words[subentities_orig['entity_class'][int(key)]]], 0)), 0)

        for key in ent_det_list:
            if word_emb_det is None:
                word_emb_det = torch.unsqueeze(vocab[int(words[subentities_det['entity_class'][int(key)]])], 0)
            else:
                word_emb_det = torch.concat((word_emb_det, torch.unsqueeze(vocab[words[subentities_det['entity_class'][int(key)]]], 0)), 0)
        
        word_emb_origt = word_emb_orig
        word_emb_dett = word_emb_det
        dist_mat = torch.cdist(word_emb_origt, word_emb_dett, p=2)
        with open(f'{data_path2}{dataset}_Imagedetails.npy','rb') as fr:
            fb2d = np.load(fr, allow_pickle=True).item()

        while len(dist_mat) > 0:
            if dist_mat.size()[1] == 0:
                for i in range(word_emb_origt.size()[0]):
                    dist_ind = torch.linalg.norm(word_emb_origt[i].unsqueeze(0).unsqueeze(1)-word_emb_det.unsqueeze(0), dim=-1)
                    arg = torch.argmin(dist_ind,dim=1)
                    if dataset=="fb15k" or dataset=="fb15k-237":
                        id2id[ent2id['/'+d[:-4].replace('.','/')+'_'+subentities_orig['entity_class'][int(ent_orig_list[int(torch.where((word_emb_orig == word_emb_origt[i]).all(dim=1))[0])])]]] = arg
                    elif dataset=="YAGO15K" or dataset=="DB15K":
                        id2id[ent2id[fb2d[d[:-4]]+'_'+subentities_orig['entity_class'][int(ent_orig_list[int(torch.where((word_emb_orig == word_emb_origt[i]).all(dim=1))[0])])]]] = arg

                break
            idx_pos = torch.argmin(dist_mat, keepdim=True)
            idx = np.array(np.unravel_index(idx_pos, dist_mat.shape)).flatten()


            if dataset=="fb15k" or dataset=="fb15k-237":
                id2id[ent2id['/'+d[:-4].replace('.','/')+'_'+subentities_orig['entity_class'][int(ent_orig_list[int(torch.where((word_emb_orig == word_emb_origt[idx[0]]).all(dim=1))[0])])]]] = torch.where((word_emb_det == word_emb_dett[idx[1]]).all(dim=1))[0]
            elif dataset=="YAGO15K" or dataset=="DB15K":
                id2id[ent2id[fb2d[d[:-4]]+'_'+subentities_orig['entity_class'][int(ent_orig_list[int(torch.where((word_emb_orig == word_emb_origt[idx[0]]).all(dim=1))[0])])]]] = torch.where((word_emb_det == word_emb_dett[idx[1]]).all(dim=1))[0]

            col = np.arange(dist_mat.shape[1])
            col = col[~np.isin(col, idx[1])]
            row = np.arange(dist_mat.shape[0])
            row = row[~np.isin(row, idx[0])]
            dist_mat = dist_mat[:,col]
            dist_mat = dist_mat[row,:]
            word_emb_dett = word_emb_dett[col,:]
            word_emb_origt = word_emb_origt[row,:]
        
        if dataset=="fb15k" or dataset=="fb15k-237":
            id2id_arr[ent2id['/'+d[:-4].replace('.','/')]] = id2id
        elif dataset == "YAGO15K" or dataset == "DB15K":
            id2id_arr[ent2id[fb2d[d[:-4]]]] = id2id

    return id2id_arr

np.save('id2id_arr_db15k.npy',dataload())
