import pickle
from tqdm import tqdm
import numpy as np
from copy import deepcopy

loc = 'dataset_baseline/FB15k-betae'

def convert(mode, answer_type):
    with open(f'{loc}/{mode}-{answer_type}-agg.pkl','rb') as fr:
        mm_answers = pickle.load(fr)
    _dict = {'tp':'easy','fn':'hard'}

    mm_answers_baseline = {}
    mm_answers_baseline2 = {}


    if mode=='train':
        for key, value in mm_answers.items():
            if not(len(value[1])==0):
                mm_answers_baseline[key] = value[1]
            mm_answers_baseline2[key] = np.concatenate((value[0],value[1]))
        
        with open(f'{loc}/{mode}-process-womm-answers.pkl','wb') as fw:
            pickle.dump(mm_answers_baseline, fw)

        with open(f'{loc}/{mode}-process-answers.pkl','wb') as fw:
            pickle.dump(mm_answers_baseline2, fw)

    elif mode=='valid':
        for key, value in mm_answers.items():
            if not(len(value[1])==0):
                mm_answers_baseline[key] = value[1]
            mm_answers_baseline2[key] = np.concatenate((value[0],value[1]))
        
        with open(f'{loc}/{mode}-{_dict[answer_type]}-process-womm-answers.pkl','wb') as fw:
            pickle.dump(mm_answers_baseline, fw)

        with open(f'{loc}/{mode}-{_dict[answer_type]}-process-answers.pkl','wb') as fw:
            pickle.dump(mm_answers_baseline2, fw)
    
    elif mode=='test':
        print(type(mm_answers))
        for key, value in mm_answers.items():
            if not(len(value[0])==0):
                mm_answers_baseline[key] = value[0]
            else:
                mm_answers_baseline2[key] = value[1]
        
        with open(f'{loc}/{mode}-{_dict[answer_type]}-process-mm-answers.pkl','wb') as fw:
            pickle.dump(mm_answers_baseline, fw)

        with open(f'{loc}/{mode}-{_dict[answer_type]}-process-womm-answers.pkl','wb') as fw:
            pickle.dump(mm_answers_baseline2, fw)

    
    if not(mode=='test'):
        with open(f'{loc}/{mode}-queries-agg.pkl','rb') as fr:
            mm_queries = pickle.load(fr)
    
        mm_queries_baseline = {}

        for key, value in mm_queries.items():
            value_dup = deepcopy(value)
            for i in value:
                if i not in mm_answers_baseline.keys():
                    value_dup.remove(i)
            mm_queries_baseline[key] = value_dup
    

        with open(f'{loc}/{mode}-queries-agg2.pkl','wb') as fw:
            pickle.dump(mm_queries_baseline, fw)

    else:
        with open(f'{loc}/{mode}-queries-agg.pkl','rb') as fr:
            mm_queries = pickle.load(fr)
    
        mm_queries_baseline = {}
        mm_queries_baseline2 = {}

        for key, value in mm_queries.items():
            value_dup = deepcopy(value)
            value_dup2 = deepcopy(value)
            for i in value:
                if i not in mm_answers_baseline.keys():
                    value_dup.remove(i)
                if i not in mm_answers_baseline2.keys():
                    value_dup2.remove(i)
            mm_queries_baseline[key] = value_dup
            mm_queries_baseline2[key] = value_dup2
    

        with open(f'{loc}/{mode}-queries-agg-mm.pkl','wb') as fw:
            pickle.dump(mm_queries_baseline, fw)
        
        with open(f'{loc}/{mode}-queries-agg-womm.pkl','wb') as fw:
            pickle.dump(mm_queries_baseline2, fw)




convert('test','tp')
convert('test','fn')
convert('valid','tp')
convert('valid','fn')
convert('train','fn')

