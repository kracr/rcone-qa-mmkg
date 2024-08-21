from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import networkx as nx

from models import KGReasoning
from dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
from util import flatten_query, parse_time, set_global_seed, eval_tuple

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
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(name_query_dict.keys())  # ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2in', '3in', 'inp', 'pin', 'pni', '2u-DNF', '2u-DM', 'up-DNF', 'up-DM']


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing ConE',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--tag', type=str, default=None, help="tag of the exp")
    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--do_train', action='store_true', help="do train")
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")

    parser.add_argument('--data_path', type=str, default=None, help="KG data path")
    parser.add_argument('--data', type=str, default=None, help="KG data")
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int,
                        help="negative entities sampled per query")
    parser.add_argument('-d', '--hidden_dim', default=500, type=int, help="embedding dimension")
    parser.add_argument('-g', '--gamma', default=12.0, type=float, help="margin in the loss")
    parser.add_argument('-b', '--batch_size', default=1024, type=int, help="batch size of queries")
    parser.add_argument('--drop', type=float, default=0., help='dropout rate')
    parser.add_argument('--test_batch_size', default=1, type=int, help='valid/test batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int, help="used to speed up torch.dataloader")
    parser.add_argument('-save', '--save_path', default=None, type=str,
                        help="no need to set manually, will configure automatically")
    parser.add_argument('--max_steps', default=100000, type=int, help="maximum iterations to train")
    parser.add_argument('--warm_up_steps', default=None, type=int,
                        help="no need to set manually, will configure automatically")

    parser.add_argument('--save_checkpoint_steps', default=1000, type=int, help="save checkpoints every xx steps")
    parser.add_argument('--valid_steps', default=10000, type=int, help="evaluate validation queries every xx steps")
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--print_on_screen', action='store_true')

    parser.add_argument('--tasks', default='1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up', type=str,
                        help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('-cenr', '--center_reg', default=0.02, type=float,
                        help='center_reg for ConE, center_reg balances the in_cone dist and out_cone dist')
    parser.add_argument('-cenr2', '--center_reg2', default=1, type=float,
                        help='center_reg2 for ConE, center_reg2 balances the mid_cone dist and out_cone dist')
    parser.add_argument('-loss_c', '--loss_const', default=1.0, type=float,
                        help='loss constant for ConE, it balances the weight of normal samles and subentity samples')
    parser.add_argument('--prefix', default=None, type=str, help='prefix of the log path')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path for loading the checkpoints')
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('-evu', '--evaluate_union', default="DNF", type=str, choices=['DNF', 'DM'],
                        help='the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')
    parser.add_argument('-tr_prob', '--train_multimodal_prob', default=1, type=float,
                        help='Probability to pick multimodal entity during training')
    parser.add_argument('-tt_prob', '--test_multimodal_prob', default=1, type=float,
                        help='Probability to pick multimodal entity during testing')
    parser.add_argument('-neigh_hop', '--neighbour_hop', default=1, type=int,
                        help='neighbourhoop hops for a multimodal entity to feed in the transformer, to get structural/semantic knowledge')

    return parser.parse_args(args)


def save_model(model, optimizer, save_variable_list, args, _iter):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler1_state_dict': None, 
        'scheduler2_state_dict': None}, 
        os.path.join(args.save_path, f'checkpoint_{_iter}')
    )


def set_logger(args):
    '''
    Write logs to console and log file
    '''
    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a+'
    )
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def evaluate(model, tp_answers, fn_answers, args, dataloader, query_name_dict, mode, step, writer, G, ent2id, id2ent):
    '''
    Evaluate queries in dataloader
    '''
    average_metrics = defaultdict(float)
    all_metrics = defaultdict(float)

    metrics, metrics_m = model.test_step(model, tp_answers, fn_answers, args, dataloader, query_name_dict, G, ent2id, id2ent)
    num_query_structures = 0
    num_queries = 0
    for query_structure in metrics:
        log_metrics(mode + " " + query_name_dict[query_structure], step, metrics[query_structure])
        for metric in metrics[query_structure]:
            writer.add_scalar("/".join([mode, query_name_dict[query_structure], metric]),
                              metrics[query_structure][metric], step)
            all_metrics["_".join([query_name_dict[query_structure], metric])] = metrics[query_structure][metric]
            if metric != 'num_queries':
                average_metrics[metric] += metrics[query_structure][metric]
        num_queries += metrics[query_structure]['num_queries']
        num_query_structures += 1

    for metric in average_metrics:
        average_metrics[metric] /= num_query_structures
        writer.add_scalar("/".join([mode, 'average', metric]), average_metrics[metric], step)
        all_metrics["_".join(["average", metric])] = average_metrics[metric]

    log_metrics('%s average' % mode, step, average_metrics)
    
    num_query_structures = 0
    num_queries = 0
    for query_structure in metrics_m:
        log_metrics(mode + " " + query_name_dict[query_structure], step, metrics_m[query_structure])
        for metric in metrics_m[query_structure]:
            writer.add_scalar("/".join([mode, query_name_dict[query_structure], metric]),
                              metrics_m[query_structure][metric], step)
            if metric != 'num_queries':
                average_metrics[metric] += metrics_m[query_structure][metric]
        num_queries += metrics_m[query_structure]['num_queries']
        num_query_structures += 1

    for metric in average_metrics:
        average_metrics[metric] /= num_query_structures
        writer.add_scalar("/".join([mode, 'average', metric]), average_metrics[metric], step)
        all_metrics["_".join(["average", metric])] = average_metrics[metric]

    log_metrics('%s average' % mode, step, average_metrics)

    return all_metrics




def load_data(args, tasks):

    logging.info("loading data")
    train_queries = pickle.load(open(os.path.join(args.data_path, "train-queries-agg.pkl"), 'rb'))
    train_answers = pickle.load(open(os.path.join(args.data_path, "train-fn-agg.pkl"), 'rb'))
    valid_queries = pickle.load(open(os.path.join(args.data_path, "valid-queries-agg.pkl"), 'rb'))
    valid_hard_answers = pickle.load(open(os.path.join(args.data_path, "valid-fn-agg.pkl"), 'rb'))
    valid_easy_answers = pickle.load(open(os.path.join(args.data_path, "valid-tp-agg.pkl"), 'rb'))

    test_queries = pickle.load(open(os.path.join(args.data_path, "test-queries-agg.pkl"), 'rb'))
    test_hard_answers = pickle.load(open(os.path.join(args.data_path, "test-fn-agg.pkl"), 'rb'))
    test_easy_answers = pickle.load(open(os.path.join(args.data_path, "test-tp-agg.pkl"), 'rb'))

    entity_set = []
    triplets = []
    
    if args.data == "fb15knmm":
        G, ent2id, id2ent = None, None, None
    else:
        entity_set = []
        triplets = []

        with open(f'{args.data_path}/train_indexified.txt','r') as fr:
            for line in fr.readlines():
                x = line.split()
                entity_set.append(int(x[0]))
                entity_set.append(int(x[2].strip()))
                triplets.append(x)

        entity_set = list(set(entity_set))
        fr = open(f'{args.data_path}/ent2id.pkl','rb')
        ent2id = pickle.load(fr)
        fr = open(f'{args.data_path}/id2ent.pkl','rb')
        id2ent = pickle.load(fr)
        G = nx.Graph()
        G = G.to_undirected()
        G.add_nodes_from(entity_set)
        for row in triplets:
            G.add_edge(int(row[0]),int(row[2]),relation=int(row[1]))



    for name in all_tasks:
        if 'u' in name:
            name, evaluate_union = name.split('-')
        else:
            evaluate_union = args.evaluate_union

        if name not in tasks or evaluate_union != args.evaluate_union:
            query_structure = name_query_dict[name if 'u' not in name else '-'.join([name, evaluate_union])]
            if query_structure in train_queries:
                del train_queries[query_structure]
            if query_structure in valid_queries:
                del valid_queries[query_structure]
            if query_structure in test_queries:
                del test_queries[query_structure]

    t1=[]
    for query_structure in train_queries:
        if (query_structure[0] == 'e'):
            for queries in train_queries[query_structure]:
                t1.append(queries[0])
        else:
            for queries in train_queries[query_structure]:
                t1.append(queries[0][0])
                t1.append(queries[1][0])



    return train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers, G, ent2id, id2ent



def main(args):

    set_global_seed(args.seed)
    tasks = args.tasks.split('.')

    if not args.tag:
        cur_time = parse_time()
    else:
        cur_time = args.tag

    if args.prefix is None:
        prefix = 'logs'
    else:
        prefix = args.prefix

    args.save_path = os.path.join(prefix, args.data_path.split('/')[-1], args.tasks)

    tmp_str = "g-{}-mode-{}-{}".format(args.gamma, args.center_reg, args.center_reg2)

    if args.checkpoint_path is not None:
        args.save_path = args.checkpoint_path
    else:
        #print(args.save_path)
        #print(tmp_str)
        #print(cur_time)
        args.save_path = os.path.join(args.save_path, tmp_str, cur_time)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print("logging to", args.save_path)
    if not args.do_train:  # if not training, then create tensorboard files in some tmp location
        writer = SummaryWriter('./logs-debug/unused-tb')
    else:
        writer = SummaryWriter(args.save_path)
    set_logger(args)

    with open('%s/stats.txt' % args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])

    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('-------------------------------' * 3)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    logging.info('#max steps: %d' % args.max_steps)
    logging.info('Evaluate unoins using: %s' % args.evaluate_union)

    train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers, G, ent2id, id2ent = load_data(
        args, tasks)

    logging.info("Training info:")
    if args.do_train:
        print("--Training Data loading Starts--")
        for query_structure in train_queries:
            logging.info(query_name_dict[query_structure] + ": " + str(len(train_queries[query_structure])))
        train_path_queries = defaultdict(set)
        train_other_queries = defaultdict(set)

        path_list = ['1p', '2p', '3p']
        for query_structure in train_queries:
            if query_name_dict[query_structure] in path_list:
                train_path_queries[query_structure] = train_queries[query_structure]
            else:
                train_other_queries[query_structure] = train_queries[query_structure]
        train_path_queries = flatten_query(train_path_queries)
        train_path_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDataset(train_path_queries, nentity, nrelation, args.negative_sample_size, train_answers, args.train_multimodal_prob),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.cpu_num,
            collate_fn=TrainDataset.collate_fn
        ))

        if len(train_other_queries) > 0:
            train_other_queries = flatten_query(train_other_queries)
            train_other_iterator = SingledirectionalOneShotIterator(DataLoader(
                TrainDataset(train_other_queries, nentity, nrelation, args.negative_sample_size, train_answers, args.train_multimodal_prob),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.cpu_num,
                collate_fn=TrainDataset.collate_fn
            ))
        else:
            train_other_iterator = None

    logging.info("Validation info:")
    if args.do_valid:
        for query_structure in valid_queries:
            logging.info(query_name_dict[query_structure] + ": " + str(len(valid_queries[query_structure])))
        valid_queries = flatten_query(valid_queries)
        valid_dataloader = DataLoader(
            TestDataset(
                valid_queries,
                args.nentity,
                args.nrelation,
                valid_hard_answers,
                args.test_multimodal_prob,
            ),
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num,
            collate_fn=TestDataset.collate_fn
        )

    logging.info("Test info:")
    if args.do_test:
        for query_structure in test_queries:
            logging.info(query_name_dict[query_structure] + ": " + str(len(test_queries[query_structure])))
        test_queries = flatten_query(test_queries)
        test_dataloader = DataLoader(
            TestDataset(
                test_queries,
                args.nentity,
                args.nrelation,
                test_hard_answers,
                args.test_multimodal_prob,
            ),
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num,
            collate_fn=TestDataset.collate_fn
        )

    model = KGReasoning(
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        use_cuda=args.cuda,
        center_reg=args.center_reg,
        center_reg2=args.center_reg2,
        neighbour_hop=args.neighbour_hop,
        loss_const=args.loss_const,
        test_batch_size=args.test_batch_size,
        query_name_dict=query_name_dict,
        drop=args.drop,
        train_ans = train_answers,
        dataset = args.data,
    )

    num_params = 0
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_params += np.prod(param.size())
    logging.info('Parameter Number: %d' % num_params)

    if args.cuda:
        model = model.cuda()

    if args.do_train:
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_learning_rate
        )
        warm_up_steps = args.max_steps // 2

    if args.checkpoint:
        logging.info('Loading checkpoint %s...' % args.save_path)
        latest_file = sorted([int(d.split('_')[1]) for d in os.listdir(args.save_path) if 'checkpoint' in d])[-1]
        checkpoint = torch.load(os.path.join(args.save_path, f'checkpoint_{latest_file}'))
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])

        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Randomly Initializing ConE Model...')
        init_step = 0

    step = init_step
    logging.info('tasks = %s' % args.tasks)
    logging.info('init_step = %d' % init_step)
    if args.do_train:
        logging.info('Start Training...')
        logging.info('learning_rate = %d' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)

    if args.do_train:
        training_logs = []
        for step in tqdm(range(init_step, args.max_steps)):
            if step == 2 * args.max_steps // 3:
                args.valid_steps *= 4


            log = model.train_step(model, optimizer, train_path_iterator, args, step, G, ent2id, id2ent)
            if step % 200 == 0:
                for metric in log:
                    writer.add_scalar('path/' + metric, log[metric], step)
            if train_other_iterator is not None:
                log = model.train_step(model, optimizer, train_other_iterator, args, step, G, ent2id, id2ent)
                if step % 200 == 0:
                    for metric in log:
                        writer.add_scalar('other/' + metric, log[metric], step)
                log = model.train_step(model, optimizer, train_path_iterator, args, step, G, ent2id, id2ent)

            training_logs.append(log)
            

            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(model, optimizer, save_variable_list, args, step)

            if step % args.valid_steps == 0 and step > 0:
                if args.do_valid:
                    logging.info('Evaluating on Valid Dataset...')
                    valid_all_metrics = evaluate(model, valid_easy_answers, valid_hard_answers, args, valid_dataloader,
                                                 query_name_dict, 'Valid', step, writer, G, ent2id, id2ent)

                if args.do_test:
                    logging.info('Evaluating on Test Dataset...')
                    test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader,
                                                query_name_dict, 'Test', step, writer, G, ent2id, id2ent)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)

                log_metrics('Training average', step, metrics)
                training_logs = []

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(model, optimizer, save_variable_list, args, step)

    try:
        print(step)
    except:
        step = 0

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict,
                                    'Test', step, writer, G, ent2id, id2ent)

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        valid_all_metrics = evaluate(model, valid_easy_answers, valid_hard_answers, args, valid_dataloader, query_name_dict,
                                    'Valid', step, writer, G, ent2id, id2ent)

    logging.info("Training finished!!")
    writer.close()


if __name__ == '__main__':
    main(parse_args())

