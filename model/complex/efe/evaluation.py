import operator
import sklearn
import sklearn.metrics
import numpy as np
import torch
from .tools import *
from tqdm import tqdm

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

class Result(object):
	"""
	Store one test results
	"""

	def __init__(self, preds, true_vals, ranks, raw_ranks, multiple_sets, dataset_path, idx):
                self.preds = preds
                self.ranks = ranks
                self.raw_ranks = raw_ranks
                self.multiple_sets = multiple_sets

		#Test if not all the prediction are the same, sometimes happens with overfitting,
		#and leads scikit-learn to output incorrect average precision (i.e ap=1)
                #print("In results")
                if not multiple_sets:
                        
                        self.true_vals = true_vals.values
                        if not (preds == preds[0]).all() :
                                #Due to the use of np.isclose in sklearn.metrics.ranking._binary_clf_curve (called by following metrics function),
                                #I have to rescale the predictions if they are too small:
                                preds_rescaled = preds

                                diffs = np.diff(np.sort(preds))
                                min_diff = min(abs(diffs[np.nonzero(diffs)]))
                                if min_diff < 1e-8 : #Default value of absolute tolerance of np.isclose
                                        preds_rescaled = (preds * ( 1e-7 / min_diff )).astype('d')

                                self.ap = sklearn.metrics.average_precision_score(self.true_vals,preds_rescaled)
                                self.precision, self.recall, self.thresholds = sklearn.metrics.precision_recall_curve(self.true_vals,preds_rescaled) 
                        else:
                                logger.warning("All prediction scores are equal, probable overfitting, replacing scores by random scores")
                                self.ap = (self.true_vals == 1).sum() / float(len(self.true_vals))
                                self.thresholds = preds[0]
                                self.precision = (self.true_vals == 1).sum() / float(len(self.true_vals))
                                self.recall = 0.5
		
		
                        self.mrr =-1
                        self.raw_mrr =-1
                        
                        if ranks is not None:
                                self.mrr = np.mean(1.0 / ranks)
                                self.raw_mrr = np.mean(1.0 / raw_ranks)
                else:
                    self.true_vals = []

                    for i in range(len(preds)):

                        #self.true_vals.append(true_vals[i].values)
                        if not (preds[i] == preds[i][0]).all() :
                            
                                preds_rescaled = preds[i]

                                torch.save({'preds':preds[i]}, f'{dataset_path}result_doc/results_{idx}')
                        else:
                                logger.warning("All prediction scores are equal, probable overfitting, replacing scores by random scores")
                                print("All prediction scores are equal, probable overfitting, replacing scores by random scores")
                                np.save(f'results_{idx}.npy', {'preds':preds[i]})
                                self.precision = (self.true_vals[i] == 1).sum() / float(len(self.true_vals[i]))
                                self.recall = 0.5
                    




class CV_Results(object):
	"""
	Class that stores predictions and scores by indexing them by model, embedding_size and lmbda
	"""

	def __init__(self):
		self.res = {}
		self.nb_params_used = {} #Indexed by model_s and embedding sizes, in order to plot with respect to the number of parameters of the model


	def add_res(self, res, model_s, embedding_size, lmbda, nb_params):
		if model_s not in self.res:
			self.res[model_s] = {}
		if embedding_size not in self.res[model_s]:
			self.res[model_s][embedding_size] = {}
		if lmbda not in self.res[model_s][embedding_size]:
			self.res[model_s][embedding_size][lmbda] = []

		self.res[model_s][embedding_size][lmbda].append( res )

		if model_s not in self.nb_params_used:
			self.nb_params_used[model_s] = {}
		self.nb_params_used[model_s][embedding_size] = nb_params


	def extract_sub_scores(self, idxs):
		"""
		Returns a new CV_Results object with scores only at the given indexes
		"""

		new_cv_res = CV_Results()

		for j, (model_s, cur_res) in enumerate(self.res.items()):
			for i,(k, lmbdas) in enumerate(cur_res.items()):
				for lmbda, res_list in lmbdas.items():
					for res in res_list:
						if res.ranks is not None:
							#Concat idxs on ranks as subject and object ranks are concatenated in a twice larger array
							res = Result(res.preds[idxs], res.true_vals[idxs], res.ranks[np.concatenate((idxs,idxs))], res.raw_ranks[np.concatenate((idxs,idxs))])
						else:
							res = Result(res.preds[idxs], res.true_vals[idxs], None, None)
						
						new_cv_res.add_res(res, model_s, k, lmbda, self.nb_params_used[model_s][k])

		return new_cv_res


	def _get_best_mean_ap(self, model_s, embedding_size):
		"""
		Averaging runs for each regularization value, and picking the best AP
		"""

		lmbdas = self.res[model_s][embedding_size]

		mean_aps = []
		var_aps = []
		for lmbda_aps in lmbdas.values():
			mean_aps.append( np.mean( [ result.ap for result in lmbda_aps] ) )
			var_aps.append( np.std( [ result.ap for result in lmbda_aps] ) )
		cur_aps_moments = zip(mean_aps, var_aps)

		return max(cur_aps_moments, key = operator.itemgetter(0)) #max by mean






	def print_MRR_and_hits_given_params(self, model_s, rank, lmbda):

		mrr = np.mean( [ res.mrr for res in self.res[model_s][rank][lmbda] ] )
		raw_mrr = np.mean( [ res.raw_mrr for res in self.res[model_s][rank][lmbda] ] )

		ranks_list = [ res.ranks for res in self.res[model_s][rank][lmbda]]
		hits_at1 = np.mean( [ (np.sum(ranks <= 1) + 1e-10) / float(len(ranks)) for ranks in ranks_list] )
		hits_at3 = np.mean( [ (np.sum(ranks <= 3) + 1e-10) / float(len(ranks)) for ranks in ranks_list] )
		hits_at10= np.mean( [ (np.sum(ranks <= 10) + 1e-10) / float(len(ranks))  for ranks in ranks_list] )

		logger.info("%s\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%i\t%f" %(model_s, mrr, raw_mrr, hits_at1, hits_at3, hits_at10, rank, lmbda))

		return ( mrr, raw_mrr, hits_at1, hits_at3, hits_at10)


	def print_MRR_and_hits(self):

		metrics = {}
	
		logger.info("Model\t\t\tMRR\tRMRR\tH@1\tH@3\tH@10\trank\tlmbda")

		for j, (model_s, cur_res) in enumerate(self.res.items()):

			best_mrr = -1.0
			for i,(k, lmbdas) in enumerate(cur_res.items()):

				mrrs = []
				for lmbda, res_list in lmbdas.items():
					mrrs.append( (lmbda, np.mean( [ result.mrr for result in res_list] ), np.mean( [ result.raw_mrr for result in res_list] ) ) )

				lmbda_mrr = max(mrrs, key = operator.itemgetter(1))
				mrr = lmbda_mrr[1];print(mrr);print(best_mrr)
				if mrr >= best_mrr:
					best_mrr = mrr
					best_raw_mrr = lmbda_mrr[2]
					best_lambda = lmbda_mrr[0]
					best_rank = k
					

			metrics[model_s] = (best_rank, best_lambda) + self.print_MRR_and_hits_given_params(model_s, best_rank, best_lambda)
		
		return metrics


		




class Scorer(object):

	def __init__(self, train, valid, test, compute_ranking_scores = False,):

		self.compute_ranking_scores = compute_ranking_scores

		self.known_obj_triples = {}
		self.known_sub_triples = {}
		if self.compute_ranking_scores:
			self.update_known_triples_dicts(train.indexes)
			#self.update_known_triples_dicts(test.indexes)
			if valid is not None:
				self.update_known_triples_dicts(valid.indexes)


	def update_known_triples_dicts(self,triples):
		for i,j,k in triples:
			if (i,j) not in self.known_obj_triples:
				self.known_obj_triples[(i,j)] = [k]
			elif k not in self.known_obj_triples[(i,j)]:
				self.known_obj_triples[(i,j)].append(k)

			if (j,k) not in self.known_sub_triples:
				self.known_sub_triples[(j,k)] = [i]
			elif i not in self.known_sub_triples[(j,k)]:
				self.known_sub_triples[(j,k)].append(i)


	def compute_scores(self, model, model_s, params, eval_set, folder=None, entity_dict=None, relation_dict=None, multiple_sets=False, dataset_path=None):

                ranks = None
                raw_ranks = None; 
                if multiple_sets: 
                        # print(eval_set)
                        for idx, i in enumerate(tqdm(eval_set)):
                            print(i)
                            test_triples, _, _ =  load_triples_from_txt([folder + f'test_doc/{i}'], entities_indexes = entity_dict, relations_indexes = relation_dict,
                                        add_sameas_rel = False, parse_line = parse_line, pt_file=True)
                            test = Triplets_set(np.array(list(test_triples.keys())), np.array(list(test_triples.values())))
                            preds = []; 

                            preds.append(model.predict(test.indexes))
                            #preds.append(model.predict(eval_set[i].indexes))
                            Result(preds, eval_set, ranks, raw_ranks, multiple_sets, dataset_path, i) 
                        return
                else: 
                        preds = model.predict(eval_set.indexes);print('compute_scores',eval_set.indexes);print(preds);

                
                if self.compute_ranking_scores:
                        #Then we compute the rank of each test:
                        nb_test = len( eval_set.values) #1000
                        ranks = np.empty( 2 * nb_test)
                        raw_ranks = np.empty(2 * nb_test)

                        if model_s.startswith("DistMult") or model_s.startswith("Complex") or model_s.startswith("CP") or model_s.startswith("TransE") or model_s.startswith("Rescal"):
                                #Fast super-ugly filtered metrics computation for Complex, DistMult, RESCAL and TransE
                                logger.info("Fast MRRs")

                                def cp_eval_o(i,j):
                                        return (u[i,:] * v[j,:]).dot(w.T)
                                def cp_eval_s(j,k):
                                        return u.dot(v[j,:] * w[k,:])
                                def distmult_eval_o(i,j):
                                    return (e[i,:] * r[j,:]).dot(e.T)
                                def distmult_eval_s(j,k):
                                    return e.dot(r[j,:] * e[k,:])
                                def complex_eval_o(i,j):
                                    return (e1[i,:] * r1[j,:]).dot(e1.T) + (e2[i,:] * r1[j,:]).dot(e2.T) + (e1[i,:] * r2[j,:]).dot(e2.T) - (e2[i,:] * r2[j,:]).dot(e1.T)
                                def complex_eval_s(j,k):
                                    return e1.dot(r1[j,:] * e1[k,:]) + e2.dot(r1[j,:] * e2[k,:]) + e1.dot(r2[j,:] * e2[k,:]) - e2.dot(r2[j,:] * e1[k,:])
                                def transe_l2_eval_o(i,j):
                                    return - np.sum(np.square((e[i,:] + r[j,:]) - e ),1)
                                def transe_l2_eval_s(j,k):
                                    return - np.sum(np.square(e + (r[j,:] - e[k,:]) ),1)
                                def transe_l1_eval_o(i,j):
                                    return - np.sum(np.abs((e[i,:] + r[j,:]) - e ),1)
                                def transe_l1_eval_s(j,k):
                                    return - np.sum(np.abs(e + (r[j,:] - e[k,:]) ),1)
                                def rescal_eval_o(i,j):
                                    return (e[i,:].dot(r[j,:,:])).dot(e.T)
                                def rescal_eval_s(j,k):
                                    return e.dot(r[j,:,:].dot(e[k,:]))
				
                                if model_s.startswith("DistMult"):
                                    e = model.e.get_value(borrow=True)
                                    r = model.r.get_value(borrow=True)
                                    eval_o = distmult_eval_o
                                    eval_s = distmult_eval_s
                                elif model_s.startswith("CP"):
                                    u = model.u.get_value(borrow=True)
                                    v = model.v.get_value(borrow=True)
                                    w = model.w.get_value(borrow=True)
                                    eval_o = cp_eval_o
                                    eval_s = cp_eval_s
                                elif model_s.startswith("Complex"):
                                    e1 = model.e1.get_value(borrow=True)
                                    r1 = model.r1.get_value(borrow=True)
                                    e2 = model.e2.get_value(borrow=True)
                                    r2 = model.r2.get_value(borrow=True)
                                    eval_o = complex_eval_o
                                    eval_s = complex_eval_s
                                elif model_s == "TransE_L2_Model":
                                    e = model.e.get_value(borrow=True)
                                    r = model.r.get_value(borrow=True)
                                    eval_o = transe_l2_eval_o
                                    eval_s = transe_l2_eval_s
                                elif model_s == "TransE_L1_Model":
                                    e = model.e.get_value(borrow=True)
                                    r = model.r.get_value(borrow=True)
                                    eval_o = transe_l1_eval_o
                                    eval_s = transe_l1_eval_s
                                elif model_s.startswith("Rescal"):
                                    e = model.e.get_value(borrow=True)
                                    r = model.r.get_value(borrow=True)
                                    eval_o = rescal_eval_o
                                    eval_s = rescal_eval_s
                        else:
                            #Generic version to compute ranks given any model:
                            logger.info("Slow MRRs")
                            n_ent = max(model.n,model.l)
                            idx_obj_mat = np.empty((n_ent,3), dtype=np.int64)
                            idx_sub_mat = np.empty((n_ent,3), dtype=np.int64)
                            idx_obj_mat[:,2] = np.arange(n_ent)
                            idx_sub_mat[:,0] = np.arange(n_ent)
                            
                            def generic_eval_o(i,j):
                                idx_obj_mat[:,:2] = np.tile((i,j),(n_ent,1))
                                return model.predict(idx_obj_mat)
                            
                            def generic_eval_s(j,k):
                                idx_sub_mat[:,1:] = np.tile((j,k),(n_ent,1))
                                return model.predict(idx_sub_mat)
                            
                            eval_o = generic_eval_o
                            eval_s = generic_eval_s


                        for a,(i,j,k) in enumerate(eval_set.indexes[:nb_test,:]):
                            #Computing objects ranks
                            res_obj = eval_o(i,j)
                            raw_ranks[a] = 1 + np.sum( res_obj > res_obj[k] )
                            ranks[a] = raw_ranks[a] -  np.sum( res_obj[self.known_obj_triples[(i,j)]] > res_obj[k] )
                            
                            #Computing subjects ranks
                            res_sub = eval_s(j,k)
                            raw_ranks[nb_test + a] = 1 + np.sum( res_sub > res_sub[i] )
                            ranks[nb_test + a] = raw_ranks[nb_test + a] - np.sum( res_sub[self.known_sub_triples[(j,k)]] > res_sub[i] )
                


                return Result(preds, eval_set, ranks, raw_ranks, multiple_sets, dataset_path) 

