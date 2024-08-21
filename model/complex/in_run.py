import scipy.io
import os
import efe
from efe.exp_generators import *
import efe.tools as tools
import collections
import argparse
from collections.abc import MutableMapping
from preprocessing import sg_graph_dict
from tqdm import tqdm
from multiprocessing import Process

collections.Callable = collections.abc.Callable

def runner(sg_location, dataset_name, entity):
            image_list = sorted([d for d in os.listdir(f'{sg_location}/{dataset_name}/{entity}/') if not d.startswith('.')])
            fb15kexp = build_data(name = f'{sg_location}/{dataset_name}/{entity}/{image_list[0]}', number_of_test_files = 0, path = f'{sg_location}/{dataset_name}/{entity}/{image_list[0]}/', inside=True, save_path = f'{sg_location}/complex_embeddings/{dataset_name}/{entity}.npy');print(fb15kexp)


	    #SGD hyper-parameter;
            params = Parameters(learning_rate = 0.5, 
						max_iter = 70, 
						batch_size = int(len(fb15kexp.train.values) / 5),  #Make 100 batches
						neg_ratio = 10, 
						valid_scores_every = 50,
						learning_rate_policy = 'adagrad',
						contiguous_sampling = False )
            all_params = { "Complex_Logistic_Model" : params } ; emb_size = 200; lmbda =0.01


            tools.logger.info( "Learning rate: " + str(params.learning_rate))
            tools.logger.info( "Max iter: " + str(params.max_iter))
            tools.logger.info( "Generated negatives ratio: " + str(params.neg_ratio))
            tools.logger.info( "Batch size: " + str(params.batch_size))

	    #Then call a local grid search, here only with one value of rank and regularization
            fb15kexp.grid_search_on_all_models(all_params, embedding_size_grid = [emb_size], lmbda_grid = [lmbda], nb_runs = 1, no_test=True, dataset_path=f'{sg_location}/complex_embeddings/{dataset_name}/models/{entity}_')


	    #Save ComplEx embeddings (last trained model, not best on grid search if multiple embedding sizes and lambdas)
            e1 = fb15kexp.models["Complex_Logistic_Model"][0].e1.get_value(borrow=True)
            e2 = fb15kexp.models["Complex_Logistic_Model"][0].e2.get_value(borrow=True)
            r1 = fb15kexp.models["Complex_Logistic_Model"][0].r1.get_value(borrow=True)
            r2 = fb15kexp.models["Complex_Logistic_Model"][0].r2.get_value(borrow=True)
            scipy.io.savemat(f'{sg_location}/complex_embeddings/{dataset_name}/{entity}.mat', \
			{'entities_real' : e1, 'relations_real' : r1, 'entities_imag' : e2, 'relations_imag' : r2  })

            return


if __name__=="__main__":
	#Load data, ensure that data is at path: 'path'/'name'/[train|valid|test].txt

        parser = argparse.ArgumentParser()
        parser.add_argument('--sg_loc', type=str)
        parser.add_argument('--dataset', type=str)
        args = parser.parse_args()

        sg_location = args.sg_loc
        dataset_name = args.dataset
        
        entity_list = [d for d in os.listdir(f'{sg_location}/{dataset_name}/') if os.path.isdir(f'{sg_location}/{dataset_name}/{d}')]

        if not os.path.exists(f'{sg_location}/complex_embeddings/{dataset_name}/models/'):
            os.makedirs(f'{sg_location}/complex_embeddings/{dataset_name}/models/')
        
        process_c = 40

        for i in tqdm(range(0, len(entity_list), process_c)):
            p = []
            for j in range(process_c):
                p.append(Process(target=runner, args = (sg_location, dataset_name, entity_list[i+j], )))
                p[-1].start()

            for j in range(process_c):
                p[j].join()

