import scipy.io
import os
import efe
from efe.exp_generators import *
import efe.tools as tools
import collections
import argparse
from collections.abc import MutableMapping

collections.Callable = collections.abc.Callable


if __name__ =="__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, 
                    help='path to the dataset')
    parser.add_argument('--dataset', type=str,
                    help='dataset name')
    parser.add_argument('--notest', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--onlytest', default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    dataset_path = args.path
    dataset_ = args.dataset

    if not os.path.exists(f'{dataset_path}result_doc'):
        os.makedirs(f'{dataset_path}result_doc')

    n_test_files = sorted(os.listdir(f'{dataset_path}/test_doc/'))


	#Load data, ensure that data is at path: 'path'/'name'/[train|valid|test].txt
    fb15kexp = build_data(name = f'{dataset_}', number_of_test_files = n_test_files, path = f'{dataset_path}');print(fb15kexp)
    
    
    #SGD hyper-parameters:
    params = Parameters(learning_rate = 0.5, 
    					max_iter = 400, 
    					batch_size = int(len(fb15kexp.train.values) / 100),  #Make 100 batches
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
    fb15kexp.grid_search_on_all_models(all_params, embedding_size_grid = [emb_size], lmbda_grid = [lmbda], nb_runs = 1, dataset_path=dataset_path, no_test=args.notest, only_test=args.onlytest)
    
    
    
    
    #Save ComplEx embeddings (last trained model, not best on grid search if multiple embedding sizes and lambdas)
    e1 = fb15kexp.models["Complex_Logistic_Model"][0].e1.get_value(borrow=True)
    e2 = fb15kexp.models["Complex_Logistic_Model"][0].e2.get_value(borrow=True)
    r1 = fb15kexp.models["Complex_Logistic_Model"][0].r1.get_value(borrow=True)
    r2 = fb15kexp.models["Complex_Logistic_Model"][0].r2.get_value(borrow=True)
    scipy.io.savemat(f'{dataset_path}result_doc/complex_embeddings.mat', \
    		{'entities_real' : e1, 'relations_real' : r1, 'entities_imag' : e2, 'relations_imag' : r2  })
