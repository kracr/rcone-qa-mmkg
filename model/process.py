import os
import numpy as np
from tqdm import tqdm

dataset = 'results/48/datasets/fb15k/'
image_folder = [d for d in os.listdir(dataset) if os.path.isdir(dataset+d)]
for image in tqdm(image_folder):
    ent_arr = []
    triplet_file = os.listdir(f'{dataset}{image}/')[0]
    with open(f'{dataset}{image}/{triplet_file}/triplets.txt','r') as fr:
        for line in fr.readlines():
            triplet = line.split()
            ent_arr.append(triplet[0].strip())
            ent_arr.append(triplet[2].strip())
    
    fw = open(f'{dataset}{image}/{triplet_file}/entities.npy','wb')
    np.save(fw,set(ent_arr))

