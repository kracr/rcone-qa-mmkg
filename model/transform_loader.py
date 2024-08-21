import torch

def dataload(entity_embedding, mm_index, G, ent2id, use_cuda, neighbour_hop, dataset, id2ent, id2fb, fb2id):
    neigh_emb = dict()
    for id_ in mm_index:
        neigh_emb[int(id_)] = entity_embedding[[n for n in G.neighbors(int(id_))]]
    if dataset == 'fb15k' or dataset == 'fb15k-237':
        cand_dir = [id2ent[int(m)][1:].replace('/','.')+'.mat' for m in mm_index]
    elif dataset=="YAGO15K" or dataset == "DB15K":
        cand_dir = [id2fb[id2ent[int(m)]]+'.mat' for m in mm_index]
    else:
        cand_dir = [id2ent[int(m)]+'.mat' for m in mm_index]

    graph_emb = dict()
    for d in cand_dir:

        mat = torch.load(f'results/32/datasets/complex_embeddings/{dataset}/{d[:-4]}.pt')
        if dataset=="fb15k" or dataset=="fb15k-237":
            if use_cuda:
                graph_emb[ent2id['/'+d[:-4].replace('.','/')]] = mat.to('cuda')
            else:
                graph_emb[ent2id['/'+d[:-4].replace('.','/')]] = mat
        elif dataset=="YAGO15K" or dataset=="DB15K":
            if use_cuda:
                graph_emb[ent2id[fb2id[d[:-4]]]] = mat.to('cuda')
            else:
                graph_emb[ent2id[fb2id[d[:-4]]]] = mat


    return neigh_emb, graph_emb

