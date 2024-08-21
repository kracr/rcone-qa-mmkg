from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformation import ViT
from transform_loader import dataload
import os


pi = 3.14159265358979323846


def convert_to_fuzzy(x, arg_embeddings):
    y = torch.tanh(2 * x) * (pi / 2) + (pi/2) - arg_embeddings
    return y


def convert_to_arg(x):
    y = torch.tanh(2 * x) * (pi / 2) + (pi / 2)
    return y


def convert_to_axis(x):
    y = torch.tanh(x) * pi
    return y


class AngleScale:
    def __init__(self, embedding_range):
        self.embedding_range = embedding_range

    def __call__(self, axis_embedding, scale=None):
        if scale is None:
            scale = pi
        return axis_embedding / self.embedding_range * scale


class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma, train_ans, test_batch_size=1, use_cuda=False, query_name_dict=None, center_reg=None, center_reg2=None, neighbour_hop=1, loss_const=None, drop=0., dataset=None):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.train_ans = train_ans
        self.loss_const = loss_const
        self.use_cuda = use_cuda
        self.neighbour_hop = neighbour_hop
        self.dataset = dataset
        if self.dataset == "YAGO15K":
            with open(f'../dataset/YAGO15K/YAGO15K_Imagedetails.npy','rb') as fr:
                self.fb2id = np.load(fr, allow_pickle=True).item()
            with open(f'../dataset/YAGO15K/YAGO15K_Imagedetailsr.npy','rb') as fr:
                self.id2fb = np.load(fr, allow_pickle=True).item()
            self.id2id_arr = np.load('id2id_arr_yago15k.npy', allow_pickle=True).item()
        elif self.dataset == "DB15K":
            with open(f'../dataset/DB15K/DB15K_Imagedetails.npy','rb') as fr:
                self.fb2id = np.load(fr, allow_pickle=True).item()
            with open(f'../dataset/DB15K/DB15K_Imagedetailsr.npy','rb') as fr:
                self.id2fb = np.load(fr, allow_pickle=True).item()
            self.id2id_arr = np.load('id2id_arr_db15k.npy', allow_pickle=True).item()
        elif self.dataset == "fb15k-237":
            self.id2id_arr = np.load('id2id_arr_fb15k-237.npy', allow_pickle=True).item()
            self.fb2id = None
            self.id2fb = None
        else:
            self.id2id_arr = np.load('id2id_arrthsm.npy', allow_pickle=True).item()
            self.fb2id = None
            self.id2fb = None

        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda() if self.use_cuda else torch.arange(
            nentity).to(torch.float).repeat(test_batch_size, 1) 
        self.query_name_dict = query_name_dict

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        self.cen = center_reg
        self.cen2 = center_reg2

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim),
                                             requires_grad=True)  # axis for entities
        self.angle_scale = AngleScale(self.embedding_range.item())  # scale axis embeddings to [-pi, pi]

        self.modulus = nn.Parameter(torch.Tensor([0.5 * self.embedding_range.item()]), requires_grad=True)
        
        self.axis_scale = 1.0
        self.arg_scale = 1.0
        self.fuzzy_scale = 1.0
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.axis_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.axis_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        
        self.arg_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.arg_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        
        self.fuzzy_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
        nn.init.uniform_(
            tensor=self.fuzzy_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.cone_proj = ConeProjection(self.entity_dim, 2400, 4)
        self.cone_intersection = ConeIntersection(self.entity_dim, drop)
        self.cone_negation = ConeNegation()
        self.transformer = ViT(800, 40, 400, 40, 1, 800, self.use_cuda, False)

    def get_entityembeddings(self):
        return self.entity_embedding

    def transform_union_query(self, queries, query_structure):
        """
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        """
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1]  
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1),
                                 torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0] * 2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return 'e', ('r',)
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return 'e', ('r', 'r')

    def train_step(self, model, optimizer, train_iterator, args, step, G, ent2id, id2ent):
        model.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures, multimodal_indicator, positive_sample_sub = next(train_iterator)
        
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)

        for i, query in enumerate(batch_queries):  
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)

        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        positive_logit, negative_logit, subsampling_weight, _, positive_logit_sub, negative_logit_sub, _, _, _, _ = model(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, multimodal_indicator, positive_sample_sub, None, G, ent2id, id2ent, False, step)

        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        if not (len(negative_logit_sub)==0):
            negative_score_sub = F.logsigmoid(-negative_logit_sub).mean(dim=1)
            negative_sample_loss_sub = - (negative_score_sub).mean()
        else:
            negative_sample_loss_sub = 0


        n_multimodal_indicator = torch.logical_not(multimodal_indicator)
        negative_score[multimodal_indicator] = 1.0*negative_score[multimodal_indicator]
        negative_score[n_multimodal_indicator] = 1.0*negative_score[n_multimodal_indicator]
        positive_score[multimodal_indicator] = 1.0*positive_score[multimodal_indicator]
        positive_score[n_multimodal_indicator] = 1.0*positive_score[n_multimodal_indicator]

        if not (len(positive_logit_sub)==0):
            positive_score_sub = F.logsigmoid(positive_logit_sub).squeeze(dim=1)
            positive_sample_loss_sub = - (positive_score_sub).sum()
        else:
            positive_sample_loss_sub = 0
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()
        loss = ((positive_sample_loss + negative_sample_loss) / 2) + 0.005*self.loss_const*((positive_sample_loss_sub + negative_sample_loss_sub) / 2)

        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }
        return log

    def test_step(self, model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, G, ent2id, id2ent, save_result=False, save_str="", save_empty=False):
        model.eval()

        total_steps = len(test_dataloader)
        metrics_m_temp = collections.defaultdict(lambda: collections.defaultdict(int))
        metrics_temp = collections.defaultdict(lambda: collections.defaultdict(int))
        step = 0

        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structures, multimodal_indicator, positive_sample_sub, multimodal_sample in tqdm(test_dataloader):
                

                logs = collections.defaultdict(list)
                logs_m = collections.defaultdict(list)
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    multimodal_sample = multimodal_sample.cuda()
                    negative_sample = negative_sample.cuda()


                _, negative_logit, _, idxs, _, _, multimodal_logit, positive_logit_sub_sum_arr, negative_logit_sub_sum_arr, negative_logit_mm = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict, multimodal_indicator, positive_sample_sub, multimodal_sample, G, ent2id, id2ent, True)
                

                idxs = torch.tensor(idxs)
                n_multimodal_indicator = torch.logical_not(multimodal_indicator)
                queries_unflatten_nm = [queries_unflatten[i] for i in idxs[n_multimodal_indicator[idxs]]]
                query_structures_nm = [query_structures[i] for i in idxs[n_multimodal_indicator[idxs]]]
                
                query_structures_m = [query_structures[i] for i in idxs[multimodal_indicator[idxs]]]
                queries_unflatten_m = [queries_unflatten[i] for i in idxs[multimodal_indicator[idxs]]]
                argsort = torch.argsort(negative_logit[idxs[n_multimodal_indicator[idxs]]], dim=1, descending=True)
                argsort_mm = torch.argsort(negative_logit[idxs[multimodal_indicator[idxs]]], dim=1, descending=True)
                ranking_mm = argsort_mm.clone().to(torch.float)
                ranking_mm = ranking_mm.scatter_(1, argsort_mm, model.batch_entity_range) 


                for idx, (i, query, query_structure) in enumerate(zip(argsort_mm[:, 0], queries_unflatten_m, query_structures_m)):
                    hard_answer = hard_answers[query]
                    try:
                        easy_answer = easy_answers[query]
                    except:
                        easy_answer = [[]]

                    ans_all = np.union1d(easy_answer[0],hard_answer[0])

                    num_ans = len(ans_all)

                    cur_ranking = ranking_mm[idx, list(ans_all.astype(np.float_))]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    if args.cuda:
                        answer_list = torch.arange(num_ans).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_ans).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1  # filtered setting

                    mrr = torch.mean(1. / cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    logs_m[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'MRR_sub': positive_logit_sub_sum_arr[idx][0],
                        'HITS1_sub': positive_logit_sub_sum_arr[idx][1],
                        'HITS3_sub': positive_logit_sub_sum_arr[idx][2],
                        'HITS5_sub': positive_logit_sub_sum_arr[idx][3],
                    })

                ranking = argsort.clone().to(torch.float)
                ranking = ranking.scatter_(1, argsort, model.batch_entity_range) 

                for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten_nm, query_structures_nm)):
                    hard_answer = hard_answers[query]
                    try:
                        easy_answer = easy_answers[query]
                        cur_ranking = ranking[idx, list(easy_answer[1].astype(np.float_)) + list(hard_answer[1].astype(np.float_))]
                    except:
                        easy_answer = [[],[]]
                        cur_ranking = ranking[idx, list(hard_answer[1].astype(np.float_))]

                    num_hard = len(hard_answer[1])
                    num_easy = len(easy_answer[1])

                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if args.cuda:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1  # filtered setting
                    cur_ranking = cur_ranking[masks]  # only take indices that belong to the hard answers

                    mrr = torch.mean(1. / cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                    })
                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

        
                for query_structure in logs_m:
                    for metric in logs_m[query_structure][0].keys():
                        metrics_m_temp[query_structure][metric] = metrics_m_temp[query_structure][metric]+sum([log[metric] for log in logs_m[query_structure]]) 
                    metrics_m_temp[query_structure]['num_queries'] = metrics_m_temp[query_structure]['num_queries']+len(logs_m[query_structure])
        
                for query_structure in logs:
                    for metric in logs[query_structure][0].keys():
                        metrics_temp[query_structure][metric] = metrics_temp[query_structure][metric]+sum([log[metric] for log in logs[query_structure]])
                    metrics_temp[query_structure]['num_queries'] = metrics_temp[query_structure]['num_queries']+len(logs[query_structure])
                step = step+1



        metrics_m = collections.defaultdict(lambda: collections.defaultdict(int))
        metrics = collections.defaultdict(lambda: collections.defaultdict(int))

        for query_structure in metrics_temp.keys():
            for metric in metrics_temp[query_structure].keys():
                metrics[query_structure][metric] = metrics_temp[query_structure][metric]/metrics_temp[query_structure]['num_queries']
                metrics[query_structure]['num_queries'] = metrics_temp[query_structure]['num_queries']
        
        for query_structure in metrics_m_temp.keys():
            for metric in metrics_m_temp[query_structure].keys():
                metrics_m[query_structure][metric] = metrics_m_temp[query_structure][metric]/metrics_m_temp[query_structure]['num_queries']
                metrics_m[query_structure]['num_queries'] = metrics_m_temp[query_structure]['num_queries']

        return metrics, metrics_m

    def embed_query_cone(self, queries, query_structure, idx):
        all_relation_flag = True

        for ele in query_structure[-1]:
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                
                axis_entity_embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                axis_entity_embedding = self.angle_scale(axis_entity_embedding, self.axis_scale)
                axis_entity_embedding = convert_to_axis(axis_entity_embedding)

                if self.use_cuda:
                    arg_entity_embedding = torch.zeros_like(axis_entity_embedding).cuda()
                    fuzzy_entity_embedding = torch.zeros_like(axis_entity_embedding).cuda()
                else:
                    arg_entity_embedding = torch.zeros_like(axis_entity_embedding)
                    fuzzy_entity_embedding = torch.zeros_like(axis_entity_embedding)
                idx += 1

                axis_embedding = axis_entity_embedding
                arg_embedding = arg_entity_embedding
                fuzzy_embedding = fuzzy_entity_embedding
            else:
                axis_embedding, arg_embedding, fuzzy_embedding, idx = self.embed_query_cone(queries, query_structure[0], idx)

            for i in range(len(query_structure[-1])):
                # negation
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    axis_embedding, arg_embedding, fuzzy_embedding = self.cone_negation(axis_embedding, arg_embedding, fuzzy_embedding)

                # projection
                else:
                    axis_r_embedding = torch.index_select(self.axis_embedding, dim=0, index=queries[:, idx])
                    arg_r_embedding = torch.index_select(self.arg_embedding, dim=0, index=queries[:, idx])
                    fuzzy_r_embedding = torch.index_select(self.fuzzy_embedding, dim=0, index=queries[:, idx])

                    axis_r_embedding = self.angle_scale(axis_r_embedding, self.axis_scale)
                    arg_r_embedding = self.angle_scale(arg_r_embedding, self.arg_scale)
                    fuzzy_r_embedding = self.angle_scale(fuzzy_r_embedding, self.fuzzy_scale)

                    axis_r_embedding = convert_to_axis(axis_r_embedding)
                    arg_r_embedding = convert_to_axis(arg_r_embedding)
                    fuzzy_r_embedding = convert_to_axis(fuzzy_r_embedding)

                    axis_embedding, arg_embedding, fuzzy_embedding = self.cone_proj(axis_embedding, arg_embedding, fuzzy_embedding, axis_r_embedding, arg_r_embedding, fuzzy_r_embedding)
                idx += 1
        else:
            # intersection
            axis_embedding_list = []
            arg_embedding_list = []
            fuzzy_embedding_list = []
            for i in range(len(query_structure)):
                axis_embedding, arg_embedding, fuzzy_embedding, idx = self.embed_query_cone(queries, query_structure[i], idx)
                axis_embedding_list.append(axis_embedding)
                arg_embedding_list.append(arg_embedding)
                fuzzy_embedding_list.append(fuzzy_embedding)

            
            stacked_axis_embeddings = torch.stack(axis_embedding_list)
            stacked_arg_embeddings = torch.stack(arg_embedding_list)
            stacked_fuzzy_embeddings = torch.stack(fuzzy_embedding_list)
            
            axis_embedding, arg_embedding, fuzzy_embedding = self.cone_intersection(stacked_axis_embeddings, stacked_arg_embeddings,stacked_fuzzy_embeddings)
            
        return axis_embedding, arg_embedding, fuzzy_embedding, idx

    def cal_logit_cone(self, entity_embedding,  query_axis_embedding, query_arg_embedding, query_fuzzy_embedding, multimodal_indicator):

        delta1 = entity_embedding - (query_axis_embedding - query_arg_embedding)
        delta2 = entity_embedding - (query_axis_embedding + query_arg_embedding)
        
        distance2axis = torch.abs(torch.sin((entity_embedding - query_axis_embedding) / 2))
        distance_base = torch.abs(torch.sin(query_arg_embedding / 2))
        distance_combine = torch.abs(torch.sin((query_arg_embedding + query_fuzzy_embedding) / 2))
        
        
        if self.use_cuda:
            distance_in = torch.zeros(distance2axis.size(), device='cuda')
            distance_mid = torch.zeros(distance2axis.size(), device ='cuda')
            indicator_in = torch.ones(distance2axis.size(), device = 'cuda', dtype=torch.bool)
            if len(distance2axis.size())==3:
                distance = torch.zeros((distance2axis.size()[0], distance2axis.size()[1]), device ='cuda')
            else:
                distance = torch.zeros((distance2axis.size()[0], distance2axis.size()[1], distance2axis.size()[2]), device ='cuda')
        else:
            distance_in = torch.zeros(distance2axis.size())
            distance_mid = torch.zeros(distance2axis.size())
            indicator_in = torch.ones(distance2axis.size(), dtype=torch.bool)
            if len(distance2axis.size())==3:
                distance = torch.zeros((distance2axis.size()[0], distance2axis.size()[1]))
            else:
                distance = torch.zeros((distance2axis.size()[0], distance2axis.size()[1], distance2axis.size()[2]))

        

        if type(multimodal_indicator) == int:
            if multimodal_indicator == -1:              
                
                delta21 = entity_embedding - (query_axis_embedding - (query_arg_embedding + query_fuzzy_embedding))
                delta22 = entity_embedding - (query_axis_embedding + (query_arg_embedding + query_fuzzy_embedding))
                
                distance_in = torch.min(distance2axis, distance_base)
                distance_mid = torch.min(torch.min(torch.abs(torch.sin(delta1 / 2)), torch.abs(torch.sin(delta2 / 2))),torch.abs(torch.sin((query_fuzzy_embedding) / 2)))
                indicator_in = distance2axis < distance_base
                distance_mid[indicator_in] = 0.
                

                distance_out = torch.min(torch.abs(torch.sin(delta21 / 2)), torch.abs(torch.sin(delta22 / 2)))
                indicator_mid = distance2axis < distance_combine
                distance_out[indicator_mid] = 0.


                distance = torch.norm(distance_out, p=1, dim=-1) + self.cen * torch.norm(distance_in, p=1, dim=-1) + self.cen2 * torch.norm(distance_mid, p=1, dim=-1)
                logit = self.gamma - distance * self.modulus


                return logit
    
            
            elif multimodal_indicator in [-2, -3]:          
                indicator_in = distance2axis < distance_base
                indicator_out = distance2axis > distance_combine
                indicator_inc = torch.all(indicator_in, dim=-1)
                indicator_outc = torch.any(indicator_out, dim=-1)
                indicator_midc = torch.logical_not(indicator_inc | indicator_outc)
                return torch.cat((indicator_inc, indicator_midc, indicator_outc),-1).float()


        else:

            delta21 = entity_embedding - (query_axis_embedding - (query_arg_embedding + query_fuzzy_embedding))
            delta22 = entity_embedding - (query_axis_embedding + (query_arg_embedding + query_fuzzy_embedding))
            
            delta31 = entity_embedding - (query_axis_embedding - (query_arg_embedding + query_fuzzy_embedding/2))
            delta32 = entity_embedding - (query_axis_embedding + (query_arg_embedding + query_fuzzy_embedding/2))
            
            distance_in[multimodal_indicator] = torch.abs(distance2axis[multimodal_indicator]- distance_base[multimodal_indicator])
            indicator_in[multimodal_indicator] = distance2axis[multimodal_indicator] > distance_base[multimodal_indicator]
            distance_in[multimodal_indicator][indicator_in[multimodal_indicator]] = 0.
            distance_mid[multimodal_indicator] = torch.min(torch.min(torch.abs(torch.sin(delta31[multimodal_indicator] / 2)), torch.abs(torch.sin(delta32[multimodal_indicator] / 2))), torch.abs(torch.sin((query_fuzzy_embedding[multimodal_indicator] / 2)/2)))
        
            
            n_multimodal_indicator = torch.logical_not(multimodal_indicator)
            distance_in[n_multimodal_indicator] = torch.min(distance2axis[n_multimodal_indicator], distance_base[n_multimodal_indicator])
            distance_mid[n_multimodal_indicator] = torch.min(torch.min(torch.abs(torch.sin(delta1[n_multimodal_indicator] / 2)), torch.abs(torch.sin(delta2[n_multimodal_indicator] / 2))),torch.abs(torch.sin((query_fuzzy_embedding[n_multimodal_indicator]) / 2)))
            indicator_in[n_multimodal_indicator] = distance2axis[n_multimodal_indicator] < distance_base[n_multimodal_indicator]
            distance_mid[n_multimodal_indicator][indicator_in[n_multimodal_indicator]] = 0.
        
        

        distance_out = torch.min(torch.abs(torch.sin(delta21 / 2)), torch.abs(torch.sin(delta22 / 2)))
        indicator_mid = distance2axis < distance_combine
        distance_out[indicator_mid] = 0.


        distance[n_multimodal_indicator] = torch.norm(distance_out[n_multimodal_indicator], p=1, dim=-1) + self.cen * torch.norm(distance_in[n_multimodal_indicator], p=1, dim=-1) + self.cen2 * torch.norm(distance_mid[n_multimodal_indicator], p=1, dim=-1)
        distance[multimodal_indicator] = torch.norm(distance_out[multimodal_indicator], p=1, dim=-1) + self.cen2 * 0.9 * torch.norm(distance_in[multimodal_indicator], p=1, dim=-1) + self.cen * torch.norm(distance_mid[multimodal_indicator], p=1, dim=-1)
        logit = self.gamma - distance * self.modulus


        return logit
    
    def evaluate_sub(self, positive_logit_sub, negative_logit_sub):
        argsort = torch.argsort(torch.concat((positive_logit_sub, negative_logit_sub)), dim=0, descending=True).squeeze(1)
        ranking = argsort.clone().to(torch.float)
        if self.use_cuda:
            ranking = ranking.scatter_(0, argsort.squeeze(), torch.arange(len(positive_logit_sub)+len(negative_logit_sub), dtype=torch.float32, device='cuda'))
        else:
            ranking = ranking.scatter_(0, argsort.squeeze(), torch.arange(len(positive_logit_sub)+len(negative_logit_sub), dtype=torch.float32))
        num_ans = len(positive_logit_sub)
        
        cur_ranking = ranking[torch.arange(num_ans)]
        cur_ranking, indices = torch.sort(cur_ranking)
        if self.use_cuda:
            answer_list = torch.arange(num_ans).to(torch.float).cuda()
        else:
            answer_list = torch.arange(num_ans).to(torch.float)
        cur_ranking = cur_ranking - answer_list + 1  # filtered setting

        mrr = torch.mean(1. / cur_ranking).item()
        h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
        h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
        h5 = torch.mean((cur_ranking <= 5).to(torch.float)).item()

        if self.use_cuda:
            return torch.tensor([[mrr, h1, h3,h5]], device='cuda')
        else:
            return torch.tensor([[mrr, h1, h3,h5]])

    
    
    
    
    
    def sub_sample_embedder(self, positive_sample_sub, mm_id, multimodal_indicator, positive_sample_regular, all_axis_embeddings, all_arg_embeddings, all_fuzzy_embeddings, sub_eval, G, ent2id, id2ent, union_indicator=False, test=False):

        positive_logit_sub_sum, negative_logit_sub_sum = None, None
        positive_logit_sub_sum2, negative_logit_sub = None, torch.Tensor([]).to(self.entity_embedding.device)
        neigh_emb, graph_emb = dataload(self.entity_embedding, mm_id, G, ent2id, self.use_cuda, self.neighbour_hop, self.dataset, id2ent, self.id2fb, self.fb2id)

        
        mm_index = multimodal_indicator == True
        for key in mm_index.nonzero():
            key = int(key)

            
            trans_embedding = self.transformer(neigh_emb[int(positive_sample_regular[key])].unsqueeze(0), graph_emb[int(positive_sample_regular[key])].unsqueeze(0), self.entity_embedding[int(positive_sample_regular[key])].unsqueeze(0)).squeeze(0)

            sub_idx = None
            sub_idx_neg = None
            for ele in positive_sample_sub[key]:
                if torch.isinf(ele):
                    break
                
                ele = int(ele)
                if ele in self.id2id_arr[int(positive_sample_regular[key])].keys():
                    if sub_idx is None:
                        sub_idx = torch.tensor([self.id2id_arr[int(positive_sample_regular[key])][ele]])
                    else:
                        sub_idx = torch.cat((sub_idx,torch.tensor([self.id2id_arr[int(positive_sample_regular[key])][ele]])))
            

            sub_idx_neg = torch.tensor(np.setdiff1d(np.arange(trans_embedding.size()[0]), np.array(sub_idx), assume_unique=True))
            
            if self.use_cuda:
                sub_idx = sub_idx.to('cuda')
                sub_idx_neg = sub_idx_neg.to('cuda')

            if union_indicator:
                positive_embedding_sub = torch.index_select(trans_embedding, dim=0, index=sub_idx).unsqueeze(1).unsqueeze(1)
            else:
                positive_embedding_sub = torch.index_select(trans_embedding, dim=0, index=sub_idx).unsqueeze(1)
            

            positive_embedding_sub = self.angle_scale(positive_embedding_sub, self.axis_scale)
            positive_embedding_sub = convert_to_axis(positive_embedding_sub)

            if (len(all_axis_embeddings.size())==2 and (not (union_indicator))) or (len(all_axis_embeddings.size())==3 and union_indicator):
                positive_logit_sub = self.cal_logit_cone(positive_embedding_sub, torch.unsqueeze(all_axis_embeddings,0), torch.unsqueeze(all_arg_embeddings,0), torch.unsqueeze(all_fuzzy_embeddings,0), sub_eval)
            else:
                positive_logit_sub = self.cal_logit_cone(positive_embedding_sub, torch.unsqueeze(all_axis_embeddings[key],0), torch.unsqueeze(all_arg_embeddings[key],0), torch.unsqueeze(all_fuzzy_embeddings[key],0), sub_eval)
            
            if union_indicator:
                positive_logit_sub = torch.max(positive_logit_sub, dim=1)[0]

            if positive_logit_sub_sum is None:
                positive_logit_sub_sum = torch.mean(positive_logit_sub, 0).unsqueeze(0)
            else:
                positive_logit_sub_sum = torch.cat((positive_logit_sub_sum, torch.mean(positive_logit_sub, 0).unsqueeze(0)))


            if not (len(sub_idx_neg) == 0):
                
                if union_indicator:
                    negative_embedding_sub = torch.index_select(trans_embedding, dim=0, index=sub_idx_neg).unsqueeze(1).unsqueeze(1)
                else:
                    negative_embedding_sub = torch.index_select(trans_embedding, dim=0, index=sub_idx_neg).unsqueeze(1)

                negative_embedding_sub = self.angle_scale(negative_embedding_sub, self.axis_scale)
                negative_embedding_sub = convert_to_axis(negative_embedding_sub)

                if (len(all_axis_embeddings.size())==2 and (not (union_indicator))) or (len(all_axis_embeddings.size())==3 and union_indicator):
                    negative_logit_sub = self.cal_logit_cone(negative_embedding_sub, torch.unsqueeze(all_axis_embeddings,0), torch.unsqueeze(all_arg_embeddings,0), torch.unsqueeze(all_fuzzy_embeddings,0), sub_eval)
                else:
                    negative_logit_sub = self.cal_logit_cone(negative_embedding_sub, torch.unsqueeze(all_axis_embeddings[key],0), torch.unsqueeze(all_arg_embeddings[key],0), torch.unsqueeze(all_fuzzy_embeddings[key],0), sub_eval)
            
                if union_indicator:
                    negative_logit_sub = torch.max(negative_logit_sub, dim=1)[0]
                
                if negative_logit_sub_sum is None:
                    negative_logit_sub_sum = torch.mean(negative_logit_sub,0).unsqueeze(0)
                else:
                    negative_logit_sub_sum = torch.cat((negative_logit_sub_sum, torch.mean(negative_logit_sub,0).unsqueeze(0)))
            
            if test:
                if positive_logit_sub_sum2 is None:
                    positive_logit_sub_sum2 = self.evaluate_sub(positive_logit_sub, negative_logit_sub)
                else:
                    positive_logit_sub_sum2 = torch.cat((positive_logit_sub_sum2, self.evaluate_sub(positive_logit_sub, negative_logit_sub)), dim=0)

        if test:
            return positive_logit_sub_sum2
        if positive_logit_sub_sum is None:
            positive_logit_sub_sum = torch.Tensor([]).to(self.entity_embedding.device)
        if negative_logit_sub_sum is None:
            negative_logit_sub_sum = torch.Tensor([]).to(self.entity_embedding.device)



        return positive_logit_sub_sum, negative_logit_sub_sum



    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, multimodal_indicator, positive_sample_sub, multimodal_sample, G, ent2id, id2ent, test, step=0):
        all_idxs, all_axis_embeddings, all_arg_embeddings, all_fuzzy_embeddings = [], [], [], []
        all_union_idxs, all_union_axis_embeddings, all_union_arg_embeddings, all_union_fuzzy_embeddings = [], [], [], []

        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                axis_embedding, arg_embedding, fuzzy_embedding, _ = \
                    self.embed_query_cone(self.transform_union_query(batch_queries_dict[query_structure], query_structure), self.transform_union_structure(query_structure), 0)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_axis_embeddings.append(axis_embedding)
                all_union_arg_embeddings.append(arg_embedding)
                all_union_fuzzy_embeddings.append(fuzzy_embedding)
            else:
                axis_embedding, arg_embedding, fuzzy_embedding, _ = self.embed_query_cone(batch_queries_dict[query_structure], query_structure, 0)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_axis_embeddings.append(axis_embedding)
                all_arg_embeddings.append(arg_embedding)
                all_fuzzy_embeddings.append(fuzzy_embedding)

        if len(all_axis_embeddings) > 0:
            all_axis_embeddings = torch.cat(all_axis_embeddings, dim=0).unsqueeze(1)
            all_arg_embeddings = torch.cat(all_arg_embeddings, dim=0).unsqueeze(1)
            all_fuzzy_embeddings = torch.cat(all_fuzzy_embeddings, dim=0).unsqueeze(1)
        if len(all_union_axis_embeddings) > 0:
            all_union_axis_embeddings = torch.cat(all_union_axis_embeddings, dim=0).unsqueeze(1)
            all_union_arg_embeddings = torch.cat(all_union_arg_embeddings, dim=0).unsqueeze(1)
            all_union_fuzzy_embeddings = torch.cat(all_union_fuzzy_embeddings, dim=0).unsqueeze(1)

            all_union_axis_embeddings = all_union_axis_embeddings.view(
                all_union_axis_embeddings.shape[0] // 2, 2, 1, -1)
            all_union_arg_embeddings = all_union_arg_embeddings.view(
                all_union_arg_embeddings.shape[0] // 2, 2, 1, -1)
            all_union_fuzzy_embeddings = all_union_fuzzy_embeddings.view(
                all_union_fuzzy_embeddings.shape[0] // 2, 2, 1, -1)
        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]


        positive_logit_sub_sum_arr, negative_logit_sub_sum_arr, positive_logit_sub_sum_arr_union, negative_logit_sub_sum_arr_union, positive_logit_sub_sum, negative_logit_sub_sum, negative_logit_mm = None, None, None, None, None, None, None

        if type(positive_sample) != type(None):
        
            if len(all_axis_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                multimodal_indicator_regular = multimodal_indicator[all_idxs]
                
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)

                positive_embedding = self.angle_scale(positive_embedding, self.axis_scale)
                positive_embedding = convert_to_axis(positive_embedding)

                positive_logit = self.cal_logit_cone(positive_embedding, all_axis_embeddings, all_arg_embeddings, all_fuzzy_embeddings, multimodal_indicator_regular)
                
                mm_id = positive_sample_regular[multimodal_indicator_regular==True]
                positive_logit_sub_sum, negative_logit_sub_sum = self.sub_sample_embedder(positive_sample_sub[all_idxs], mm_id, multimodal_indicator_regular, positive_sample_regular, all_axis_embeddings, all_arg_embeddings, all_fuzzy_embeddings, -1, G, ent2id, id2ent)

            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)
                positive_logit_sub_sum = torch.Tensor([]).to(self.entity_embedding.device)
                negative_logit_sub_sum = torch.Tensor([]).to(self.entity_embedding.device)



            if len(all_union_axis_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                multimodal_indicator_union = multimodal_indicator[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)

                positive_embedding = self.angle_scale(positive_embedding, self.axis_scale)
                positive_embedding = convert_to_axis(positive_embedding)

                positive_union_logit = self.cal_logit_cone(positive_embedding, all_union_axis_embeddings,all_union_arg_embeddings, all_union_fuzzy_embeddings, multimodal_indicator_union)


                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
                mm_id_union = positive_sample_union[multimodal_indicator_union==True]
                positive_logit_sub_sum_union, negative_logit_sub_sum_union = self.sub_sample_embedder(positive_sample_sub[all_union_idxs], mm_id_union, multimodal_indicator_union, positive_sample_union, all_union_axis_embeddings, all_union_arg_embeddings, all_union_fuzzy_embeddings, -1, G, ent2id, id2ent, True)
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
                positive_logit_sub_sum_union = torch.Tensor([]).to(self.entity_embedding.device)
                negative_logit_sub_sum_union = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
            positive_logit_sub_sum = torch.cat([positive_logit_sub_sum, positive_logit_sub_sum_union], dim=0) 
            negative_logit_sub_sum = torch.cat([negative_logit_sub_sum, negative_logit_sub_sum_union], dim=0) 
        
        else:
            positive_logit = None
        
        if (not type(multimodal_sample) == type(None)):
        
            if len(all_axis_embeddings) > 0: 
                multimodal_sample_regular = multimodal_sample[all_idxs]
                positive_sample_sub_regular = positive_sample_sub[all_idxs]
                multimodal_sample_regular = multimodal_sample_regular[torch.any(multimodal_sample_regular<float("Inf"), 1)]
                positive_sample_sub_regular = positive_sample_sub_regular[torch.any(positive_sample_sub_regular<float("Inf"), 1)]
                multimodal_logit = None
                for l in range(len(multimodal_sample_regular)):
                    mm_id = multimodal_sample_regular[l][multimodal_sample_regular[l]<float("Inf")]
                        
                    positive_logit_sub_sum = self.sub_sample_embedder(positive_sample_sub_regular[l].repeat(len(mm_id),1), mm_id, multimodal_sample_regular[l]<float("Inf"), multimodal_sample_regular[l], all_axis_embeddings[l], all_arg_embeddings[l], all_fuzzy_embeddings[l], -1, G, ent2id, id2ent, test=True)
                    if positive_logit_sub_sum_arr is None:
                        positive_logit_sub_sum_arr = torch.sum(positive_logit_sub_sum, 0).unsqueeze(0)/len(positive_logit_sub_sum)
                    else:
                        positive_logit_sub_sum_arr = torch.cat((positive_logit_sub_sum_arr, torch.sum(positive_logit_sub_sum, 0).unsqueeze(0)/len(positive_logit_sub_sum)))


            else:
                multimodal_logit = torch.Tensor([]).to(self.entity_embedding.device)
            if multimodal_logit is None:
                multimodal_logit = torch.Tensor([]).to(self.entity_embedding.device)
            if positive_logit_sub_sum_arr is None:
                positive_logit_sub_sum_arr = torch.Tensor([]).to(self.entity_embedding.device)
            if negative_logit_sub_sum_arr is None:
                negative_logit_sub_sum_arr = torch.Tensor([]).to(self.entity_embedding.device)



            if len(all_union_axis_embeddings) > 0: 
                multimodal_sample_union = multimodal_sample[all_union_idxs]
                positive_sample_sub_union = positive_sample_sub[all_union_idxs]
                multimodal_sample_union = multimodal_sample_union[torch.any(multimodal_sample_union<float("Inf"), 1)]
                positive_sample_sub_union = positive_sample_sub_union[torch.any(positive_sample_sub_union<float("Inf"), 1)]

                multimodal_logit_union = None
                for l in range(len(multimodal_sample_union)):
                    mm_id_union = multimodal_sample_union[l][multimodal_sample_union[l]<float("Inf")]
                    positive_logit_sub_sum_union = self.sub_sample_embedder(positive_sample_sub_union[l].repeat(len(mm_id_union),1), mm_id_union, multimodal_sample_union[l]<float("Inf"), multimodal_sample_union[l], all_union_axis_embeddings[l], all_union_arg_embeddings[l], all_union_fuzzy_embeddings[l], -1, G, ent2id, id2ent, True, test=True)
                    if positive_logit_sub_sum_arr_union is None:
                        positive_logit_sub_sum_arr_union = torch.sum(positive_logit_sub_sum_union, 0).unsqueeze(0)/len(positive_logit_sub_sum_union)
                    else:
                        positive_logit_sub_sum_arr_union = torch.cat((positive_logit_sub_sum_arr_union, torch.sum(positive_logit_sub_sum_union, 0).unsqueeze(0)/len(positive_logit_sub_sum_union)))

            else:
                multimodal_logit_union = torch.Tensor([]).to(self.entity_embedding.device)
            if positive_logit_sub_sum_arr_union is None:
                positive_logit_sub_sum_arr_union = torch.Tensor([]).to(self.entity_embedding.device)
            if negative_logit_sub_sum_arr_union is None:
                negative_logit_sub_sum_arr_union = torch.Tensor([]).to(self.entity_embedding.device)

            if multimodal_logit_union is None:
                multimodal_logit_union = torch.Tensor([]).to(self.entity_embedding.device)
            multimodal_logit = torch.cat([multimodal_logit, multimodal_logit_union], dim=0)
            positive_logit_sub_sum_arr = torch.cat([positive_logit_sub_sum_arr, positive_logit_sub_sum_arr_union], dim=0)
            negative_logit_sub_sum_arr = torch.cat([negative_logit_sub_sum_arr, negative_logit_sub_sum_arr_union], dim=0)

        else:
            multimodal_logit = None
        



        if type(negative_sample) != type(None):
            if len(all_axis_embeddings) > 0:
                #print(6)
                negative_sample_regular = negative_sample[all_idxs]
                multimodal_indicator_regular = multimodal_indicator[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_embedding = self.angle_scale(negative_embedding, self.axis_scale)
                negative_embedding = convert_to_axis(negative_embedding)

                if not test:
                    negative_logit = self.cal_logit_cone(negative_embedding, all_axis_embeddings, all_arg_embeddings, all_fuzzy_embeddings, torch.zeros(multimodal_indicator_regular.size(), dtype=torch.bool))
                else:
                    negative_logit = self.cal_logit_cone(negative_embedding, all_axis_embeddings, all_arg_embeddings, all_fuzzy_embeddings, multimodal_indicator_regular)
                    
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)
                if test:
                    positive_logit_sub_sum = torch.Tensor([]).to(self.entity_embedding.device)
                    negative_logit_sub_sum = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_axis_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                multimodal_indicator_union = multimodal_indicator[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1)
                negative_embedding = self.angle_scale(negative_embedding, self.axis_scale)
                negative_embedding = convert_to_axis(negative_embedding)

                if not test:
                    negative_union_logit = self.cal_logit_cone(negative_embedding, all_union_axis_embeddings, all_union_arg_embeddings,all_union_fuzzy_embeddings, torch.zeros(multimodal_indicator_union.size(), dtype=torch.bool))
                else:
                    negative_union_logit = self.cal_logit_cone(negative_embedding, all_union_axis_embeddings, all_union_arg_embeddings,all_union_fuzzy_embeddings, multimodal_indicator_union)

                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)

        else:
            negative_logit = None



        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs, positive_logit_sub_sum, negative_logit_sub_sum, multimodal_logit, positive_logit_sub_sum_arr, negative_logit_sub_sum_arr, negative_logit_mm



class ConeProjection(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers):
        super(ConeProjection, self).__init__()
        self.axis_dim = dim
        self.arg_dim = dim
        self.fuzzy_dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.axis_dim + self.arg_dim + self.fuzzy_dim, self.hidden_dim)  
        self.layer0 = nn.Linear(self.hidden_dim, self.axis_dim + self.arg_dim + self.fuzzy_dim)  
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

    def forward(self, source_embedding_axis, source_embedding_arg, source_embedding_fuzzy, r_embedding_axis, r_embedding_arg, r_embedding_fuzzy):
        x = torch.cat([source_embedding_axis + r_embedding_axis,
            source_embedding_arg + r_embedding_arg,
            source_embedding_fuzzy + r_embedding_fuzzy], dim=-1)

        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)

        axis, arg, fuzzy = torch.chunk(x, 3, dim=-1)
        axis_embeddings = convert_to_axis(axis)
        arg_embeddings = convert_to_arg(arg)
        fuzzy_embeddings = convert_to_fuzzy(fuzzy,arg_embeddings)

        return axis_embeddings, arg_embeddings, fuzzy_embeddings


class ConeIntersection(nn.Module):
    def __init__(self, dim, drop):
        super(ConeIntersection, self).__init__()
        self.dim = dim
        self.layer_axis13 = nn.Linear(self.dim * 4, self.dim)
        self.layer_arg13 = nn.Linear(self.dim * 4, self.dim)
        self.layer_fuzzy13 = nn.Linear(self.dim * 4, self.dim)
        self.layer_axis2 = nn.Linear(self.dim, self.dim)
        self.layer_arg2 = nn.Linear(self.dim, self.dim)
        self.layer_fuzzy2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer_axis13.weight)
        nn.init.xavier_uniform_(self.layer_arg13.weight)
        nn.init.xavier_uniform_(self.layer_fuzzy13.weight)
        nn.init.xavier_uniform_(self.layer_axis2.weight)
        nn.init.xavier_uniform_(self.layer_arg2.weight)
        nn.init.xavier_uniform_(self.layer_fuzzy2.weight)
        
        self.drop = nn.Dropout(p=drop)

    def forward(self, axis_embeddings, arg_embeddings, fuzzy_embeddings):
        
        logits = torch.cat([axis_embeddings - arg_embeddings, axis_embeddings + arg_embeddings, axis_embeddings - (arg_embeddings + fuzzy_embeddings), axis_embeddings + (arg_embeddings + fuzzy_embeddings)], dim=-1)


        axis_layer1_act = F.relu(self.layer_axis13(logits))

        axis_attention = F.softmax(self.layer_axis2(axis_layer1_act), dim=0)

        x_embeddings = torch.cos(axis_embeddings)
        y_embeddings = torch.sin(axis_embeddings)
        x_embeddings = torch.sum(axis_attention * x_embeddings, dim=0)
        y_embeddings = torch.sum(axis_attention * y_embeddings, dim=0)

        x_embeddings[torch.abs(x_embeddings) < 1e-3] = 1e-3

        axis_embeddings = torch.atan(y_embeddings / x_embeddings)

        indicator_x = x_embeddings < 0
        indicator_y = y_embeddings < 0
        indicator_two = indicator_x & torch.logical_not(indicator_y)
        indicator_three = indicator_x & indicator_y

        axis_embeddings[indicator_two] = axis_embeddings[indicator_two] + pi
        axis_embeddings[indicator_three] = axis_embeddings[indicator_three] - pi
        
        arg_layer1_act = F.relu(self.layer_arg13(logits))

        arg_layer1_mean = torch.mean(arg_layer1_act, dim=0)
        gate = torch.sigmoid(self.layer_arg2(arg_layer1_mean))

        arg_embeddings_out = self.drop(arg_embeddings)
        arg_embeddings_out, _ = torch.min(arg_embeddings_out, dim=0)
        arg_embeddings_out = arg_embeddings_out * gate
        

        
        #Fuzzy set
        fuzzy_layer1_act = F.relu(self.layer_fuzzy13(logits))

        fuzzy_layer1_mean = torch.mean(fuzzy_layer1_act, dim=0)
        gate_fuzzy = torch.sigmoid(self.layer_fuzzy2(fuzzy_layer1_mean))

        
        fuzzy_embeddings = self.drop(fuzzy_embeddings)
        max_fuzzy, _ = torch.max(fuzzy_embeddings, dim=0)
        min_fuzzy_sum, _ =  torch.min(fuzzy_embeddings + arg_embeddings, dim=0)
        fuzzy_embeddings = torch.min(max_fuzzy, min_fuzzy_sum-arg_embeddings_out)
        fuzzy_embeddings = fuzzy_embeddings * gate_fuzzy

        return axis_embeddings, arg_embeddings_out, fuzzy_embeddings


class ConeNegation(nn.Module):
    def __init__(self):
        super(ConeNegation, self).__init__()

    def forward(self, axis_embedding, arg_embedding, fuzzy_embedding):
        indicator_positive = axis_embedding >= 0
        indicator_negative = axis_embedding < 0

        axis_embedding[indicator_positive] = axis_embedding[indicator_positive] - pi
        axis_embedding[indicator_negative] = axis_embedding[indicator_negative] + pi

        arg_embedding = pi - (arg_embedding + fuzzy_embedding)

        return axis_embedding, arg_embedding, fuzzy_embedding
