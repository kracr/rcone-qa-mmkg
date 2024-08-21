from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import random
from torch.utils.data import Dataset
from util import list2tuple, tuple2list, flatten
import itertools


def pad_tensors(*args):
    return torch.tensor(list(itertools.zip_longest(*args, fillvalue=float("Inf")))).T

class TestDataset(Dataset):
    def __init__(self, queries, nentity, nrelation, answer, test_multimodal_prob):
        self.len = len(queries)
        self.queries = queries
        self.nentity = nentity
        self.nrelation = nrelation
        self.answer = answer
        self.test_multimodal_prob = test_multimodal_prob

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        negative_sample = torch.LongTensor(range(self.nentity))
        if not (len(self.answer[query][0]) == 0) and random.random() <= self.test_multimodal_prob: 
            multimodal_indicator = torch.BoolTensor([True])
            multimodal_sample = torch.tensor(self.answer[query][0].astype(np.float_))
            positive_sample_sub = torch.tensor(self.answer[query][2].astype(np.float_))
        else:
            multimodal_indicator = torch.BoolTensor([False])
            positive_sample_sub = torch.tensor([float("Inf")])
            multimodal_sample = torch.tensor([float("Inf")])

        return negative_sample, flatten(query), query, query_structure, multimodal_indicator, positive_sample_sub, multimodal_sample

    @staticmethod
    def collate_fn(data):
        negative_sample = torch.stack([_[0] for _ in data], dim=0)
        query = [_[1] for _ in data]
        query_unflatten = [_[2] for _ in data]
        query_structure = [_[3] for _ in data]
        multimodal_indicator = torch.cat([_[4] for _ in data])
        
        padded_output = pad_tensors(*[_[5] for _ in data])
        positive_sample_sub = torch.stack([x for x in padded_output], dim=0)
        
        
        padded_output = pad_tensors(*[_[6] for _ in data])
        multimodal_sample = torch.stack([x for x in padded_output], dim=0)
        
        return negative_sample, query, query_unflatten, query_structure, multimodal_indicator, positive_sample_sub, multimodal_sample


class TrainDataset(Dataset):
    def __init__(self, queries, nentity, nrelation, negative_sample_size, answer, train_multimodal_prob):
        self.len = len(queries)
        self.queries = queries
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.count = self.count_frequency(queries, answer)
        self.answer = answer
        self.train_multimodal_prob = train_multimodal_prob

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        
        positive_sample_sub = None

        if len(self.answer[query][1]) == 0:
            tail = np.random.choice(list(self.answer[query][0]))
            multimodal_indicator = torch.BoolTensor([True])
            multi = True

            positive_sample_sub = torch.LongTensor(list(self.answer[query][2]))
        elif len(self.answer[query][0]) == 0:
            tail = np.random.choice(list(self.answer[query][1]))
            multimodal_indicator = torch.BoolTensor([False])
            positive_sample_sub = torch.tensor([float("Inf")])
            multi=False
        elif random.random() <= self.train_multimodal_prob:
            tail = np.random.choice(list(self.answer[query][1]))
            multimodal_indicator = torch.BoolTensor([False])
            positive_sample_sub = torch.tensor([float("Inf")])
            multi=False
        else:
            tail = np.random.choice(list(self.answer[query][0]))
            multimodal_indicator = torch.BoolTensor([True])
            positive_sample_sub = torch.LongTensor(list(self.answer[query][2]))
            multi=True
        


        subsampling_weight = self.count[query]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        negative_sample_list = []
        negative_sample_size = 0
        if multi:
            while negative_sample_size < self.negative_sample_size:
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
                mask = np.in1d(
                    negative_sample,
                    self.answer[query][0],
                    assume_unique=True,
                    invert=True
                )
                negative_sample = negative_sample[mask]
                negative_sample_list.append(negative_sample)
                negative_sample_size += negative_sample.size
            negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
            negative_sample = torch.from_numpy(negative_sample)
        else:
            while negative_sample_size < self.negative_sample_size:
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
                mask = np.in1d(
                    negative_sample,
                    self.answer[query][1],
                    assume_unique=True,
                    invert=True
                )
                negative_sample = negative_sample[mask]
                negative_sample_list.append(negative_sample)
                negative_sample_size += negative_sample.size
            negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
            negative_sample = torch.from_numpy(negative_sample)

        positive_sample = torch.LongTensor([tail])
        return positive_sample, negative_sample, subsampling_weight, flatten(query), query_structure, multimodal_indicator, positive_sample_sub

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.cat([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        query = [_[3] for _ in data]
        query_structure = [_[4] for _ in data]
        multimodal_indicator = torch.cat([_[5] for _ in data], dim=0)
        padded_output = pad_tensors(*[_[6] for _ in data])
        positive_sample_sub = torch.stack([x for x in padded_output], dim=0)

        return positive_sample, negative_sample, subsample_weight, query, query_structure, multimodal_indicator, positive_sample_sub

    @staticmethod
    def count_frequency(queries, answer, start=4):
        count = {}
        for query, qtype in queries:
            count[query] = start + len(list(answer[query][0])) + len(list(answer[query][1]))
        return count


class SingledirectionalOneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0

    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data
