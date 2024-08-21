import pickle
import os.path as osp
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import random
from copy import deepcopy, copy
import time
import logging
import os
import argparse
from complex.preprocessing import sg_graph_dict


def set_logger(save_path, query_name, print_on_screen=False):
    '''
    Write logs to checkpoint and console
    '''

    log_file = os.path.join(save_path, '%s.log'%(query_name))

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    if print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def convert_files(data_path, file):
    
    n = file.split('queries')[0]+'tp-answers'
    name = f'{data_path}/{n}_dup.pkl'
    file_o = open(f'{name}','wb')
    y={}
    print(name)
    with open(data_path+'/'+file.split('queries')[0]+'tp-answers_final.pkl','rb') as ftp:
        x = pickle.load(ftp)
        for a in tqdm(x.keys()):
            y[a] = [np.array(list(x[a]['multimodal_answers']), dtype=np.uint16), np.array(list(x[a]['other_answers']), dtype=np.uint16), np.array(list(x[a]['subentities']), dtype=np.uint16)]
        
        pickle.dump(y, file_o)
    
    file_o.close()
    y={}
    n = file.split('queries')[0]+'fn-answers'
    name = f'{data_path}/{n}_dup.pkl'
    file_o = open(f'{name}','wb')
    with open(data_path+'/'+file.split('queries')[0]+'fn-answers_final.pkl','rb') as ffn:
        x = pickle.load(ffn)
        for a in tqdm(x.keys()):
            y[a] = [np.array(list(x[a]['multimodal_answers']), dtype=np.uint16), np.array(list(x[a]['other_answers']), dtype=np.uint16), np.array(list(x[a]['subentities']), dtype=np.uint16)]
        
        pickle.dump(y, file_o)
    
    file_o.close()
    
def post_process(query_name_dict, data_path, mode):
    keys = list(query_name_dict.keys())
    val = list(query_name_dict.values())
    files = [d for d in os.listdir(data_path) if (f'{mode}-' in d) and ('queries_final' in d) and ('-test' not in d)]
    queries = {}
    tp_answers = {}
    fn_answers = {}


    
    for f in tqdm(files):
        convert_files(data_path, f)

        with open(data_path+'/'+f,'rb') as fq:
            queries[keys[val.index(f.split('-queries')[0].split(f'{mode}-')[1])]] = pickle.load(fq)[keys[val.index(f.split('-queries')[0].split(f'{mode}-')[1])]]
        

        with open(data_path+'/'+f.split('queries')[0]+'tp-answers_dup.pkl','rb') as ftp:
            tp_answers.update(pickle.load(ftp))

        with open(data_path+'/'+f.split('queries')[0]+'fn-answers_dup.pkl','rb') as ffn:
            fn_answers.update(pickle.load(ffn))
    
    pickle.dump(queries,open(data_path+'/../'+f'{mode}-queries-agg.pkl','wb'))
    pickle.dump(tp_answers,open(data_path+'/../'+f'{mode}-tp-agg.pkl','wb'))
    pickle.dump(fn_answers,open(data_path+'/../'+f'{mode}-fn-agg.pkl','wb'))


    return

def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def index_dataset(sg_location, dataset_name, base_path, force=False, gen_valid=False, gen_test=False, query_folder=''):
    print('Indexing dataset {0}'.format(dataset_name))
    
    scene_graph_arr = sg_graph_dict(sg_location, dataset_name, base_path)
    files=['train.txt','train_connected2.txt', 'test_connected2.txt', 'valid_connected2.txt','scenegraph_triplets']
    indexified_files=['train_indexified.txt','train_connected_indexified.txt', 'test_connected_indexified.txt', 'valid_connected_indexified.txt','scenegraph_triplets_indexified']

    if gen_valid:
        files.append('valid.txt')
        indexified_files.append('valid_indexified.txt')
    if gen_test:
        files.append('test.txt')
        indexified_files.append('test_indexified.txt')
    
    ent2id, rel2id, id2rel, id2ent = {}, {}, {}, {}

    ent2idwsg, rel2idwsg = {}, {}

    entid, relid = 0, 0
    
    file_len={}
    # adding 
    for p, indexified_p in zip(files, indexified_files):
        print('Files indexified',p)
        
        if not 'scenegraph_triplets' in p:
            if not osp.exists(base_path+query_folder+'/index'):
                os.makedirs(base_path+query_folder+'/index')

            fw = open(osp.join(base_path, query_folder, indexified_p), "w")
            with open(osp.join(base_path, p), 'r') as f:
                for i, line in enumerate(f):
                    e1, rel, e2 = line.split('\t')
                    e1 = e1.strip()
                    e2 = e2.strip()
                    rel = rel.strip()
                    rel_reverse = '-' + rel
                    rel = '+' + rel

                    if p == "train_connected2.txt" or p == "train.txt" or p == "test_connected2.txt" or p=="valid_connected2.txt":
                        if e1 not in ent2id.keys():
                            ent2id[e1] = entid
                            id2ent[entid] = e1
                            entid += 1

                        if e2 not in ent2id.keys():
                            ent2id[e2] = entid
                            id2ent[entid] = e2
                            entid += 1

                        if not rel in rel2id.keys():
                            rel2id[rel] = relid
                            id2rel[relid] = rel
                            assert relid % 2 == 0
                            relid += 1

                        if not rel_reverse in rel2id.keys():
                            rel2id[rel_reverse] = relid
                            id2rel[relid] = rel_reverse
                            assert relid % 2 == 1
                            relid += 1
                    
                    else:
                        if not ((e1 in ent2id.keys()) and (e2 in ent2id.keys()) and (rel in rel2id.keys()) and (rel_reverse in rel2id.keys())):
                            
                            print("Test/Validation set has new entry")
                            exit(-1)

                    if e1 in ent2id.keys() and e2 in ent2id.keys():
                        fw.write("\t".join([str(ent2id[e1]), str(rel2id[rel]), str(ent2id[e2])]) + "\n")
                        fw.write("\t".join([str(ent2id[e2]), str(rel2id[rel_reverse]), str(ent2id[e1])]) + "\n")
                if p == "train.txt":
                    ent2idwsg = deepcopy(ent2id)
                    rel2idwsg = deepcopy(rel2id)
            fw.close()

        else:
            iter_ = 0
            for key, value in scene_graph_arr.items():
                if 'FB15k-multimodal' in base_path or 'FB15k-237-multimodal' in base_path:
                    combine_path = base_path+query_folder+'/index/'+indexified_p+'_'+str(key[1:].replace("/","."))+'.txt'
                else:
                    combine_path = base_path+query_folder+'/index/'+indexified_p+'_'+str(key)+'.txt'

                if not osp.exists(base_path+query_folder+'/index/'):
                    os.makedirs(base_path+query_folder+'/index/')
                fw = open(combine_path, "w")
                for triple in value:
                    e1 = key+'_'+str(triple[0])
                    e2 = key+'_'+str(triple[2])
                    rel = str(triple[1])
                    rel_reverse = '-' + rel
                    rel = '+' + rel

                    if e1 in ent2id.keys() and e2 in ent2id.keys():
                        fw.write("\t".join([str(ent2id[e1]), str(rel2id[rel]), str(ent2id[e2])]) + "\n")
                        fw.write("\t".join([str(ent2id[e2]), str(rel2id[rel_reverse]), str(ent2id[e1])]) + "\n")
                iter_+=1
                fw.close()
    #print(base_path)
    #print(query_folder)

    #print(osp.join(base_path, query_folder, "stats.txt"))
    with open(osp.join(base_path, query_folder, "stats.txt"), "w") as fw:
        fw.write("numentity: " + str(len(ent2idwsg)) + "\n")
        fw.write("numrelations: " + str(len(rel2id)))
    with open(osp.join(base_path, query_folder, "stats_complete.txt"), "w") as fw:
        fw.write("numentity: " + str(len(ent2id)) + "\n")
        fw.write("numrelations: " + str(len(rel2id)))
    with open(osp.join(base_path, query_folder, 'ent2id.pkl'), 'wb') as handle:
        pickle.dump(ent2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(base_path, query_folder, 'rel2id.pkl'), 'wb') as handle:
        pickle.dump(rel2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(base_path, query_folder, 'id2ent.pkl'), 'wb') as handle:
        pickle.dump(id2ent, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(base_path, query_folder, 'id2rel.pkl'), 'wb') as handle:
        pickle.dump(id2rel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print ('num entity: %d, num relation: %d'%(len(ent2id), len(rel2id)))
    print ("indexing finished!!")

def construct_graph(base_path, indexified_files):
    ent_in, ent_out = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
    for indexified_p in indexified_files:
        with open(osp.join(base_path, indexified_p)) as f:
            for i, line in enumerate(f):
                if len(line) == 0:
                    continue
                e1, rel, e2 = line.split('\t')
                e1 = int(e1.strip())
                e2 = int(e2.strip())
                rel = int(rel.strip())
                ent_out[e1][rel].add(e2)
                ent_in[e2][rel].add(e1)

    return ent_in, ent_out

def list2tuple(l):
    return tuple(list2tuple(x) if type(x)==list else x for x in l)

def tuple2list(t):
    return list(tuple2list(x) if type(x)==tuple else x for x in t)


def ho_multimodal_answer_gen(answers, multimodal_entity_dict, multimodal_subentities):
    multimodal_answers = set()
    for enti in multimodal_entity_dict.keys():
        present = False
        for subent in multimodal_entity_dict[enti]:
            if subent in answers:
                present = True
                break
        if present:
            multimodal_answers.add(enti)
    return {'all_answers': answers, 'multimodal_answers': multimodal_answers, 'other_answers': answers.difference(set(multimodal_subentities)), 'subentities': set(multimodal_subentities).intersection(answers)}
        

def write_links(dataset, ent_out, small_ent_out, max_ans_num, name, multimodal_subentities, data_path, MULTIMODAL_WRITE_LINKS, multimodal_entity_dict, mode):
    queries = defaultdict(set)
    tp_answers = defaultdict(set)
    fn_answers = defaultdict(set)
    num_more_answer = 0
    
    candidate_entities = ent_out

    if not MULTIMODAL_WRITE_LINKS:
        for e in multimodal_subentities:
            if e in candidate_entities:
                del candidate_entities[e]

    for ent in tqdm(candidate_entities):
        for rel in ent_out[ent]:
            if len(ent_out[ent][rel]) <= max_ans_num: #and len(ent_out[ent][rel].difference(multimodal_subentities))>0:
                queries[('e', ('r',))].add((ent, (rel,)))
                if mode == 'train':
                    fn_answers[(ent, (rel,))] = ho_multimodal_answer_gen(ent_out[ent][rel], multimodal_entity_dict, multimodal_subentities)
                else:
                    tp_answers[(ent, (rel,))] = ho_multimodal_answer_gen(small_ent_out[ent][rel], multimodal_entity_dict, multimodal_subentities)
                    fn_answers[(ent, (rel,))] = ho_multimodal_answer_gen(ent_out[ent][rel], multimodal_entity_dict, multimodal_subentities)
                    
                    if len(fn_answers[(ent, (rel,))]['all_answers'] - tp_answers[(ent, (rel,))]['all_answers']) == 0:
                        continue

                    fn_answers[(ent, (rel,))]['all_answers'] = fn_answers[(ent, (rel,))]['all_answers'] - tp_answers[(ent, (rel,))]['all_answers']
                    fn_answers[(ent, (rel,))]['multimodal_answers'] = fn_answers[(ent, (rel,))]['multimodal_answers'] - tp_answers[(ent, (rel,))]['multimodal_answers']
                    fn_answers[(ent, (rel,))]['other_answers'] = fn_answers[(ent, (rel,))]['other_answers'] - tp_answers[(ent, (rel,))]['other_answers']
                    fn_answers[(ent, (rel,))]['subentities'] = fn_answers[(ent, (rel,))]['subentities'] - tp_answers[(ent, (rel,))]['subentities']
                
            else:
                num_more_answer += 1

    with open(f'{data_path}/iid/%s-queries.pkl'%(name), 'wb') as f:
        pickle.dump(queries, f)
    with open(f'{data_path}/iid/%s-tp-answers.pkl'%(name), 'wb') as f:
        pickle.dump(tp_answers, f)
    with open(f'{data_path}/iid/%s-fn-answers.pkl'%(name), 'wb') as f:
        pickle.dump(fn_answers, f)
    print (num_more_answer)

def ground_queries(dataset, query_structure, ent_in, ent_out, small_ent_in, small_ent_out, gen_num, max_ans_num, query_name, mode, ent2id, rel2id, target_manswer_set, multimodal_entity_dict, multimodal_subentities, base_path, per_mm, nmm_entities, query_folder):
    num_sampled, num_try, num_repeat, num_more_answer, num_broken, num_no_extra_answer, num_no_extra_negative, num_empty = 0, 0, 0, 0, 0, 0, 0, 0
    tp_ans_num, fn_ans_num = [], []
    queries = defaultdict(set)
    tp_answers = defaultdict(set)
    fn_answers = defaultdict(set)
    s0 = time.time()
    old_num_sampled = -1
    recieved_old_ans = True
    need_mm_ans = False
    while num_sampled < gen_num:
        if num_sampled != 0:
            if num_sampled % (gen_num//100) == 0 and num_sampled != old_num_sampled:
                logging.info('%s %s: [%d/%d], avg time: %s, try: %s, repeat: %s: more_answer: %s, broken: %s, no extra: %s, no negative: %s empty: %s'%(mode, 
                    query_structure, 
                    num_sampled, gen_num, (time.time()-s0)/num_sampled, num_try, num_repeat, num_more_answer, 
                    num_broken, num_no_extra_answer, num_no_extra_negative, num_empty))
                old_num_sampled = num_sampled
        
        print ('%s %s: [%d/%d], avg time: %s, try: %s, repeat: %s: more_answer: %s, broken: %s, no extra: %s, no negative: %s empty: %s'%(mode, 
            query_structure, 
            num_sampled, gen_num, (time.time()-s0)/(num_sampled+0.001), num_try, num_repeat, num_more_answer, 
            num_broken, num_no_extra_answer, num_no_extra_negative, num_empty), end='\r')
        
        num_try += 1
        empty_query_structure = deepcopy(query_structure)
        temp_target_manswer_set = deepcopy(target_manswer_set)
        if not (recieved_old_ans):
            if need_mm_ans:
                answer = np.random.choice(temp_target_manswer_set, 1)[0]
            else:
                answer = random.randint(0, nmm_entities-1)
        else:
            recieved_old_ans = False
            if np.random.rand() < per_mm:
                answer = np.random.choice(temp_target_manswer_set, 1)[0]
                need_mm_ans = True
            else:
                answer = random.randint(0, nmm_entities-1)
                need_mm_ans = False
        answer_inbetween = copy([answer])
        start_entity_set = []
        broken_flag = fill_query(empty_query_structure, ent_in, ent_out, answer, ent2id, rel2id, answer_inbetween, multimodal_subentities, start_entity_set)
        if broken_flag:
            num_broken += 1
            continue
        query = empty_query_structure
        
        if not (len(set(start_entity_set).intersection(multimodal_subentities))==0):
            continue

        answer_set = achieve_answer(query, ent_in, ent_out)
        small_answer_set = achieve_answer(query, small_ent_in, small_ent_out)
        if len(answer_set) == 0:
            num_empty += 1
            continue
        
        if mode != 'train':
            if len(answer_set - small_answer_set) == 0:
                num_no_extra_answer += 1
                continue
            if 'n' in query_name:
                if len(small_answer_set - answer_set) == 0:
                    num_no_extra_negative += 1
                    continue
        
        if max(len(answer_set - small_answer_set), len(small_answer_set - answer_set)) > max_ans_num:
            num_more_answer += 1
            continue
        if list2tuple(query) in queries[list2tuple(query_structure)]:
            num_repeat += 1
            continue
        

        queries[list2tuple(query_structure)].add(list2tuple(query))
        recieved_old_ans = True
        if mode == 'train':
            tp_actual_answers = answer_set
            fn_actual_answers = answer_set
        else:
            fn_actual_answers = answer_set - small_answer_set
            tp_actual_answers = small_answer_set

        tp_answers[list2tuple(query)] = ho_multimodal_answer_gen(tp_actual_answers, multimodal_entity_dict, multimodal_subentities)
        fn_answers[list2tuple(query)] = ho_multimodal_answer_gen(fn_actual_answers, multimodal_entity_dict, multimodal_subentities)
        
        
        num_sampled += 1
        tp_ans_num.append(len(tp_answers[list2tuple(query)]['all_answers']))
        fn_ans_num.append(len(fn_answers[list2tuple(query)]['all_answers']))

    print ()
    logging.info ("{} tp max: {}, min: {}, mean: {}, std: {}".format(mode, np.max(tp_ans_num), np.min(tp_ans_num), np.mean(tp_ans_num), np.std(tp_ans_num)))
    logging.info ("{} fn max: {}, min: {}, mean: {}, std: {}".format(mode, np.max(fn_ans_num), np.min(fn_ans_num), np.mean(fn_ans_num), np.std(fn_ans_num)))

    name_to_save = '%s-%s'%(mode, query_name)
    with open(f'{base_path}/{query_folder}/iid/%s-queries.pkl'%(name_to_save), 'wb') as f:
        pickle.dump(queries, f)
    with open(f'{base_path}/{query_folder}/iid/%s-fn-answers.pkl'%(name_to_save), 'wb') as f:
        pickle.dump(fn_answers, f)
    with open(f'{base_path}/{query_folder}/iid/%s-tp-answers.pkl'%(name_to_save), 'wb') as f:
        pickle.dump(tp_answers, f)
    return queries, tp_answers, fn_answers

def generate_queries(dataset, query_structures, gen_num, max_ans_num, gen_train, gen_valid, gen_test, query_names, save_name, target_manswer_set_train, target_manswer_set_valid, target_manswer_set_test, multimodal_entity_dict, multimodal_subentities, idx, base_path, mm_writelink, per_mm, nmm_entities, query_folder):
    
    indexified_files = ['train_connected_indexified.txt', 'valid_connected_indexified.txt', 'test_connected_indexified.txt']
    if gen_train or gen_valid:
        train_ent_in, train_ent_out = construct_graph(base_path+query_folder+'/', indexified_files[:1]) # ent_in 
    if gen_valid or gen_test:
        valid_ent_in, valid_ent_out = construct_graph(base_path+query_folder+'/', indexified_files[:2])
        valid_only_ent_in, valid_only_ent_out = construct_graph(base_path+query_folder+'/', indexified_files[1:2])
    if gen_test:
        test_ent_in, test_ent_out = construct_graph(base_path+query_folder+'/', indexified_files[:3])
        test_only_ent_in, test_only_ent_out = construct_graph(base_path+query_folder+'/', indexified_files[2:3])

    ent2id = pickle.load(open(os.path.join(base_path, query_folder, "ent2id.pkl"), 'rb'))
    rel2id = pickle.load(open(os.path.join(base_path, query_folder,"rel2id.pkl"), 'rb'))

    train_queries = defaultdict(set)
    train_tp_answers = defaultdict(set)
    train_fn_answers = defaultdict(set)
    valid_queries = defaultdict(set)
    valid_tp_answers = defaultdict(set)
    valid_fn_answers = defaultdict(set)
    test_queries = defaultdict(set)
    test_tp_answers = defaultdict(set)
    test_fn_answers = defaultdict(set)

    assert len(query_structures) == 1
    query_structure = query_structures[0]
    query_name = query_names[idx] if save_name else str(idx)
    print ('general structure is', query_structure, "with name", query_name)
    if query_structure == ['e', ['r']]:
        if gen_train:
            write_links(dataset, train_ent_out, defaultdict(lambda: defaultdict(set)), max_ans_num, 'train-'+query_name, multimodal_subentities, base_path+query_folder+'/', mm_writelink, multimodal_entity_dict, 'train')
        if gen_valid:
            write_links(dataset, valid_only_ent_out, train_ent_out, max_ans_num, 'valid-'+query_name, multimodal_subentities, base_path+query_folder+'/', mm_writelink, multimodal_entity_dict, 'valid')
        if gen_test:
            write_links(dataset, test_only_ent_out, valid_ent_out, max_ans_num, 'test-'+query_name, multimodal_subentities, base_path+query_folder+'/', mm_writelink, multimodal_entity_dict, 'test')
        print ("link prediction created!")
        exit(-1)
    
    name_to_save = query_name
    set_logger(f'{base_path}/{query_folder}/', name_to_save)

    s0 = time.time()
    if gen_train:
        train_queries, train_tp_answers, train_fn_answers = ground_queries(dataset, query_structure, 
            train_ent_in, train_ent_out, defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set)), 
            gen_num[0], max_ans_num, query_name, 'train', ent2id, rel2id, target_manswer_set_train, multimodal_entity_dict, multimodal_subentities, base_path, per_mm, nmm_entities, query_folder)
    if gen_valid:
        valid_queries, valid_tp_answers, valid_fn_answers = ground_queries(dataset, query_structure, 
            valid_ent_in, valid_ent_out, train_ent_in, train_ent_out, gen_num[1], max_ans_num, query_name, 'valid', ent2id, rel2id, list(set(target_manswer_set_train+target_manswer_set_valid)), multimodal_entity_dict, multimodal_subentities, base_path, per_mm, nmm_entities, query_folder)
    if gen_test:
        test_queries, test_tp_answers, test_fn_answers = ground_queries(dataset, query_structure, 
            test_ent_in, test_ent_out, valid_ent_in, valid_ent_out, gen_num[2], max_ans_num, query_name, 'test', ent2id, rel2id, list(set(target_manswer_set_train+target_manswer_set_valid+target_manswer_set_test)), multimodal_entity_dict, multimodal_subentities, base_path, per_mm, nmm_entities, query_folder)
    print ('%s queries generated with structure %s'%(gen_num, query_structure))

def fill_query(query_structure, ent_in, ent_out, answer, ent2id, rel2id, answer_inbetween, multimodal_subentities, start_entity_set):
    assert type(query_structure[-1]) == list
    all_relation_flag = True
    for ele in query_structure[-1]:
        if ele not in ['r', 'n']:
            all_relation_flag = False
            break
    if all_relation_flag:
        r = -1
        for i in range(len(query_structure[-1]))[::-1]:
            if query_structure[-1][i] == 'n':
                query_structure[-1][i] = -2
                continue
            found = False

            for j in range(40):
                r_tmp = random.sample(ent_in[answer].keys(), 1)[0]
                if r_tmp // 2 != r // 2 or r_tmp == r:
                    r = r_tmp
                    found = True
                    break
            if not found:
                return True
            query_structure[-1][i] = r
            answer = random.sample(ent_in[answer][r], 1)[0]
            answer_inbetween.append(answer)
        if query_structure[0] == 'e':
            query_structure[0] = answer
            start_entity_set.append(answer)
        else:
            return fill_query(query_structure[0], ent_in, ent_out, answer, ent2id, rel2id, answer_inbetween, multimodal_subentities, start_entity_set)
    else:
        same_structure = defaultdict(list)
        for i in range(len(query_structure)):
            same_structure[list2tuple(query_structure[i])].append(i)
        for i in range(len(query_structure)):
            if len(query_structure[i]) == 1 and query_structure[i][0] == 'u':
                assert i == len(query_structure) - 1
                query_structure[i][0] = -1
                continue
            broken_flag = fill_query(query_structure[i], ent_in, ent_out, answer, ent2id, rel2id, answer_inbetween, multimodal_subentities, start_entity_set)
            if broken_flag:
                return True
        for structure in same_structure:
            if len(same_structure[structure]) != 1:
                structure_set = set()
                for i in same_structure[structure]:
                    structure_set.add(list2tuple(query_structure[i]))
                if len(structure_set) < len(same_structure[structure]):
                    return True

def achieve_answer(query, ent_in, ent_out):
    assert type(query[-1]) == list
    all_relation_flag = True
    last_nmm = set()
    for ele in query[-1]:
        if (type(ele) != int) or (ele == -1):
            all_relation_flag = False
            break
    if all_relation_flag:
        if type(query[0]) == int:
            ent_set = set([query[0]])
            last_nmm = set([query[0]])
        else:
            ent_set = achieve_answer(query[0], ent_in, ent_out)
        for i in range(len(query[-1])):
            if query[-1][i] == -2:
                ent_set = set(range(len(ent_in))) - ent_set
            else:
                ent_set_traverse = set()
                for ent in ent_set:
                    ent_set_traverse = ent_set_traverse.union(ent_out[ent][query[-1][i]])
                
                ent_set = set(ent_set_traverse)
    else:
        ent_set = achieve_answer(query[0], ent_in, ent_out)
        union_flag = False
        if len(query[-1]) == 1 and query[-1][0] == -1:
            union_flag = True
        for i in range(1, len(query)):
            if not union_flag:
                ent_set = ent_set.intersection(achieve_answer(query[i], ent_in, ent_out))
            else:
                if i == len(query) - 1:
                    continue
                ent_set = ent_set.union(achieve_answer(query[i], ent_in, ent_out))
    return ent_set


def file_handler(sg_location, dataset_name, base_path, query_folder):
    
    target_manswer_set_train = []
    target_manswer_set_valid = []
    target_manswer_set_test = []
    
    meta = np.load(base_path+'meta2.npy', allow_pickle=True).item()
    scene_graph_arr, mmtriples_train, mmtriples_valid, mmtriples_test = meta['entity_mmtriples'], meta['mmtriples_train'], meta['mmtriples_valid'], meta['mmtriples_test']

    multimodal_arr = []
    multimodal_dict = dict()


    ent2id = pickle.load(open(os.path.join(base_path, query_folder,"ent2id.pkl"), 'rb'))

    if dataset_name == "fb15k" or dataset_name == "fb15k-237":
        for key, values in scene_graph_arr.items():
            if not(key.replace('/','.')[1:] in mmtriples_train ):
                continue
            temp_mdict = []
            for triplets in values:
                temp_mdict.append(ent2id[key+'_'+triplets[0]])
                temp_mdict.append(ent2id[key+'_'+triplets[2]])
            multimodal_arr.append(list(set(temp_mdict)))
            multimodal_dict[ent2id[key]] = list(set(temp_mdict))
        
            if key.replace('/','.')[1:] in mmtriples_train:
                target_manswer_set_train.extend(temp_mdict)
            elif key.replace('/','.')[1:] in mmtriples_valid:
                target_manswer_set_valid.extend(temp_mdict)
            elif key.replace('/','.')[1:] in mmtriples_test:
                target_manswer_set_test.extend(temp_mdict)

    else:
        for key, values in scene_graph_arr.items():
            if not(key in mmtriples_train ):
                continue
            temp_mdict = []
            for triplets in values:
                temp_mdict.append(ent2id[key+'_'+triplets[0]])
                temp_mdict.append(ent2id[key+'_'+triplets[2]])
            multimodal_arr.append(list(set(temp_mdict)))
            multimodal_dict[ent2id[key]] = list(set(temp_mdict))
        
            if key in mmtriples_train:
                target_manswer_set_train.extend(temp_mdict)
            elif key in mmtriples_valid:
                target_manswer_set_valid.extend(temp_mdict)
            elif key in mmtriples_test:
                target_manswer_set_test.extend(temp_mdict)

    
    target_manswer_set_train = list(set(target_manswer_set_train))
    target_manswer_set_valid = list(set(target_manswer_set_valid))
    target_manswer_set_test = list(set(target_manswer_set_test))
    multimodal_subentities = deepcopy(list(set(target_manswer_set_train+target_manswer_set_valid+target_manswer_set_test)))

    return multimodal_dict, multimodal_subentities, target_manswer_set_train, target_manswer_set_valid, target_manswer_set_test

parser=argparse.ArgumentParser()
parser.add_argument("-g","--gen_id", type=int, default=0, help="")
parser.add_argument('--dataset', type=str, default="mFB15k")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gen_train_num', type=int, default=0)
parser.add_argument('--gen_valid_num', type=int, default=0)
parser.add_argument('--gen_test_num', type=int, default=0)
parser.add_argument('--max_ans_num', type=int, default=100)
parser.add_argument('--reindex', type=bool, default=True)
parser.add_argument('--gen_train', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--gen_valid', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--gen_test', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--save_name', type=bool, default=True)
parser.add_argument('--index_only', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--post_combine', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--sg_loc', type=str)
parser.add_argument('--path', type=str)
parser.add_argument('--per_mm', type=float)
parser.add_argument('--mm_writelink', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--query_folder', type=str)

args = parser.parse_args()
def main(args):


    query_name_dict = {('e', ('r',)): '1p',
                   ('e', ('r', 'r')): '2p',
                   ('e', ('r', 'r', 'r')): '3p',
                   (('e', ('r',)), ('e', ('r',))): '2i',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                   ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                   (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                       (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                   ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                   (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                   (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                   (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                   ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                   }

    e = 'e'
    r = 'r'
    n = 'n'
    u = 'u'
    query_structures = [
                        [e, [r]],
                        [e, [r, r]],
                        [e, [r, r, r]],
                        [[e, [r]], [e, [r]]],
                        [[e, [r]], [e, [r]], [e, [r]]],
                        [[e, [r, r]], [e, [r]]],
                        [[[e, [r]], [e, [r]]], [r]],
                        # negation
                        [[e, [r]], [e, [r, n]]],
                        [[e, [r]], [e, [r]], [e, [r, n]]],
                        [[e, [r, r]], [e, [r, n]]],
                        [[e, [r, r, n]], [e, [r]]],
                        [[[e, [r]], [e, [r, n]]], [r]],
                        # union
                        [[e, [r]], [e, [r]], [u]],
                        [[[e, [r]], [e, [r]], [u]], [r]]
                       ]
    query_names = ['1p', '2p', '3p', '2i', '3i', 'pi', 'ip', '2in', '3in', 'pin', 'pni', 'inp', '2u-DNF', 'up-DNF']


    #Initialization
    train_num_dict = {'FB15k': 273710, "FB15k-237": 149689, "NELL":
            107982, 'mFB15k': 1000}
    valid_num_dict = {'FB15k': 8000, "FB15k-237": 5000, "NELL":
            4000, 'mFB15k': 8000}
    test_num_dict = {'FB15k': 8000, "FB15k-237": 5000, "NELL":
            4000, 'mFB15k': 8000}
    if args.gen_train and args.gen_train_num == 0:
        args.gen_train_num = train_num_dict[args.dataset]
    if args.gen_valid and args.gen_valid_num == 0:
        args.gen_valid_num = valid_num_dict[args.dataset]
    if args.gen_test and args.gen_test_num == 0:
        args.gen_test_num = test_num_dict[args.dataset]
    # Indexing
    if args.index_only:
        index_dataset(args.sg_loc, args.dataset, args.path, args.reindex, args.gen_valid,
                args.gen_test, args.query_folder)
        exit(-1)
    if args.post_combine:
        post_process(query_name_dict, args.path+args.query_folder+'/iid', 'train')
        print('a')
        post_process(query_name_dict, args.path+args.query_folder+'/iid', 'valid')
        print('b')
        post_process(query_name_dict, args.path+args.query_folder+'/iid', 'test')
        exit(-1)

    with open(args.path+args.query_folder+'/stats.txt','r') as fr:
        for lines in fr.readlines():
            nmm_entities = int(lines.split(':')[1].strip())
            break

    if not osp.exists(args.path+args.query_folder+'/iid'):
        os.makedirs(args.path+args.query_folder+'/iid')
    
    multimodal_entity_dict, multimodal_subentities, target_manswer_set_train, target_manswer_set_valid, target_manswer_set_test = file_handler(args.sg_loc, args.dataset, args.path, args.query_folder)


    generate_queries(args.dataset,
            query_structures[args.gen_id:args.gen_id+1],
            [args.gen_train_num, args.gen_valid_num,
                args.gen_test_num], args.max_ans_num,
            args.gen_train, args.gen_valid, args.gen_test,
            query_names, args.save_name, target_manswer_set_train, target_manswer_set_valid, 
                     target_manswer_set_test, multimodal_entity_dict, multimodal_subentities, args.gen_id, 
                     args.path, args.mm_writelink, args.per_mm, nmm_entities, args.query_folder)

if __name__ == '__main__':
    main(args)
