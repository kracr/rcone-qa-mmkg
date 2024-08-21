import os
import torch

def dataset_prepare(dataset_name):
    
    dataset_path = f'datasets/{dataset_name}'
    
    file_name = []
    out_dir = []

    dir_list = os.listdir(dataset_path)
    
    for _dir in dir_list:
        sub_dir = dataset_path+'/'+_dir+'/'
        file_list = os.listdir(sub_dir)
        for _file in file_list:
            file_name.append(f'{sub_dir}{_file}')
            file_wf = _file.split('.')[0]
            out_dir.append(f'output/sg_graph/{sub_dir}{file_wf}')
            if not os.path.exists(out_dir[-1]):
                os.makedirs(out_dir[-1])

    return file_name, out_dir


def sg_triplets(scene_graph, out_dir):
    
    rel_ind = torch.argmax(scene_graph.rel_scores, dim=1)
    triplet = torch.tensor([[]])
        
    for i in range(scene_graph.rel_scores.size()[0]):
        if i==0:
            triplet = torch.tensor([[scene_graph.pred_classes[scene_graph.rel_inds[i][0]], rel_ind[i], scene_graph.pred_classes[scene_graph.rel_inds[i][1]]]])
        triplet = torch.cat((triplet,torch.tensor([[scene_graph.pred_classes[scene_graph.rel_inds[i][0]], rel_ind[i], scene_graph.pred_classes[scene_graph.rel_inds[i][1]]]])),0)
    triplet = torch.unique(triplet, sorted=False, dim=0)
    
    with open(f'{out_dir}/triplets.txt','w') as fw:
        for i in range(len(triplet)):
            fw.write(f'{triplet[i][0]}\t{triplet[i][1]}\t{triplet[i][2]}\n')

    return
