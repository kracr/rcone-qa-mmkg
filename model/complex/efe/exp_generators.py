import scipy
import scipy.io
#import random
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .experiment import *


def parse_line(filename, line,i, pt_file):
    if not pt_file:
        line = line.strip().split("\t")
    sub = line[0]
    rel = line[1]
    obj = line[2]
    val = 1

    return sub,obj,rel,val


def load_triples_from_txt(filenames, entities_indexes = None, relations_indexes = None, add_sameas_rel = False, parse_line = parse_line, pt_file=False):
    """
    Take a list of file names and build the corresponding dictionnary of triples
    """
    
    
    if entities_indexes is None:
        entities_indexes= dict()
        entities = set()
        next_ent = 0
    else:
        entities = set(entities_indexes)
        next_ent = max(entities_indexes.values()) + 1
    
    
    if relations_indexes is None:
        relations_indexes= dict()
        relations= set()
        next_rel = 0
    else:
        relations = set(relations_indexes)
        next_rel = max(relations_indexes.values()) + 1
    
    data = dict()
    
    
    
    for filename in filenames:
        if not pt_file:
            with open(filename) as f:
                lines = f.readlines()
        else:
            lines = torch.load(filename)
    
        for i,line in enumerate(lines):
    
            sub,obj,rel,val = parse_line(filename, line,i, pt_file)
    
    
            if sub in entities:
            	sub_ind = entities_indexes[sub]
            else:
            	sub_ind = next_ent
            	next_ent += 1
            	entities_indexes[sub] = sub_ind
            	entities.add(sub)
            	
            if obj in entities:
            	obj_ind = entities_indexes[obj]
            else:
            	obj_ind = next_ent
            	next_ent += 1
            	entities_indexes[obj] = obj_ind
            	entities.add(obj)
            	
            if rel in relations:
            	rel_ind = relations_indexes[rel]
            else:
            	rel_ind = next_rel
            	next_rel += 1
            	relations_indexes[rel] = rel_ind
            	relations.add(rel)
            
            data[ (sub_ind, rel_ind, obj_ind)] = val
    
    
    return data, entities_indexes, relations_indexes


def build_data(name, number_of_test_files, path = '/home/ttrouill/dbfactor/projects/relational_bench/datasets/', inside=False, save_path = ''):

        folder = path; test_triples, test = [], []
        
        if inside:

            train_triples, entities_indexes, relations_indexes = load_triples_from_txt([folder+'triplets.txt'], 
					add_sameas_rel = False, parse_line = parse_line)
            train = Triplets_set(np.array(list(train_triples.keys())), np.array(list(train_triples.values())))
            print(folder)
            print(entities_indexes)

            with open(save_path, 'wb') as fw:
                np.save(fw, entities_indexes)
            
            return Experiment(name,train, None, None, None, positives_only = True, compute_ranking_scores = False, entities_dict = entities_indexes, relations_dict = relations_indexes)





        else:
            train_triples, entities_indexes, relations_indexes = load_triples_from_txt([folder + 'train_withimg.txt'], 
					add_sameas_rel = False, parse_line = parse_line)
        
        
        
            train = Triplets_set(np.array(list(train_triples.keys())), np.array(list(train_triples.values())))
        
            
            return Experiment(name,train, None, number_of_test_files, folder, positives_only = True, compute_ranking_scores = False, entities_dict = entities_indexes, relations_dict = relations_indexes)




def load_mat_file(name, path, matname, load_zeros = False, prop_valid_set = .1, prop_test_set=0):

	x = scipy.io.loadmat(path + name)[matname]


	if sp.issparse(x): 
		if not load_zeros:
			idxs = x.nonzero()

			indexes = np.array(zip(idxs[0], np.zeros_like(idxs[0]), idxs[1]))
			np.random.shuffle(indexes)

			nb = indexes.shape[0]
			i_valid = int(nb - nb*prop_valid_set - nb * prop_test_set)
			i_test = i_valid + int( nb*prop_valid_set)

			train = Triplets_set(indexes[:i_valid,:], np.ones(i_valid))
			valid = Triplets_set(indexes[i_valid:i_test,:], np.ones(i_test - i_valid))
			test = Triplets_set(indexes[i_test:,:], np.ones(nb - i_test))


	return Experiment(name,train, valid, test, positives_only = True, compute_ranking_scores = True)
	




