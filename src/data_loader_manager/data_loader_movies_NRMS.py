"""
data_loader_movies.py:  
    Data loader for Content-based Filtering Models
    Loads movie dataset from data
"""

__author__ = "Weizhe Lin"
__copyright__ = "Copyright 2021, Weizhe Lin"
__version__ = "1.0.0"
__email__ = "wl356@cam.ac.uk"
__status__ = "Published for Github"

import os
import re
import sys
import time
import json
import copy
from tqdm import tqdm
import csv
import json
import torch
import pickle
from copy import deepcopy
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import random
from time import time
from datetime import datetime
from pprint import pprint
from easydict import EasyDict
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from data_loader_manager.data_loader_wrapper import DataLoaderWrapper
from utils.dirs import create_dirs
from transformers import BertTokenizerFast
from data_loader_manager.data_loader_movies import DataLoaderMovies, MovieDataset


class DataLoaderMoviesForNRMS(DataLoaderMovies):
    '''
    Data loader for our movie-user dataset
    Compactify the user interactions
    '''

    def __init__(self, config):
        DataLoaderMovies.__init__(self, config)
    
    def set_dataloader(self):
        """This function wraps datasets into dataloader for trainers
        """
        train_dataset_dict = {
            'entity_pairs': self.train_cf_pairs,
            'user_set': self.train_user_set,
            'mode': 'train',
            'n_params': self.n_params,
        }
        self.train_dataset = NRMSBERTMovieDataset(self.config, train_dataset_dict)
        # for i in self.train_dataset:
        #     print(i)
        #     input()
        train_sampler = RandomSampler(self.train_dataset)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.config.train.batch_size,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=4,
        )
        # for i in self.train_dataloader:
        #     print('batch get')
        #     input()
        test_dataset_dict = {
            'entity_pairs': self.test_cf_pairs,
            'user_set': self.test_user_set,
            'mode': 'test',
            'n_params': self.n_params,
        }
        self.test_dataset = NRMSBERTMovieDataset(self.config, test_dataset_dict)

        test_sampler = SequentialSampler(self.test_dataset)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            sampler=test_sampler,
            batch_size=self.config.test.batch_size,
            collate_fn=self.test_dataset.collate_fn,
            num_workers=4,
        )
        print('statistics: training data loader: {};  test data loader: {}'.format(
                                len(self.train_dataloader), 
                                len(self.test_dataloader)))

class NRMSBERTMovieDataset(MovieDataset):
    def __init__(self, config, data):
        MovieDataset.__init__(self, config, data)

    def __len__(self):
        return len(self.entity_pairs)

    def __getitem__(self, idx):
        def negative_sampling(user_item, num_samples=1):
            """Generate negative samples for a user.

            Args:
                user_item (int tensor): user id
                num_samples (int, optional): number of samples. Defaults to 1.

            Returns:
                neg_items: list of negative item ids.
            """
            neg_items = []
            while len(neg_items) < num_samples:
                # sample num_samples negative items for the user
                user = int(user_item.numpy())
                while True:
                    if self.p is not None:
                        neg_item = np.random.randint(low=0, high=self.n_params.n_items, size=1)[0]
                    else:
                        neg_item = np.random.choice(self.n_params.n_items, 1, p=self.p)[0]
                    if neg_item not in self.user_set[user]:
                        break
                neg_items.append(neg_item)
            return neg_items
        
        user = self.entity_pairs[idx, 0]
        pos_item = self.entity_pairs[idx, 1]
        # Sample user histories
        if not self.config.model_config.use_att_mask:
            k=self.config.model_config.num_history
            user_history = random.choices(self.user_set[int(user)], k=k)
            history_att_mask = torch.ones(k).long().numpy().tolist()
        else:
            exist_history = self.user_set[int(user)]
            k=self.config.model_config.num_history
            history_att_mask = torch.zeros(k).long()
            if len(exist_history) <= k:
                # less than needed
                user_history = exist_history + [0]*(k-len(exist_history))
                history_att_mask[:len(exist_history)] = 1
            elif len(exist_history) > k:
                # more than needed
                user_history = random.sample(exist_history, k)
                history_att_mask[:] = 1
            history_att_mask = history_att_mask.numpy().tolist()

        feed_dict = {
            'user': user,
            'pos_item': pos_item,
            'user_history': user_history,
            'user_history_att_mask': history_att_mask,
        }
        neg_items = negative_sampling(user,num_samples=self.config.model_config.num_negative_samples)
        # print(user, pos_item, neg_items)
        # currently using only the first item
        # neg_item = neg_items
        feed_dict['neg_item'] = neg_items

        return feed_dict

    
    def collate_fn(self, batch):  # optional but useful
        '''
            when collate_fn is given to the torch dataloader, we can do further actions to the batch, e.g., tensor can be formed here
            a batch is formed as a list where each element is a defined data returned by __getitem__, andy
        '''

        users = torch.LongTensor([ex['user'] for ex in batch])
        pos_items = torch.LongTensor([ex['pos_item'] for ex in batch])
        user_histories = torch.LongTensor([ex['user_history'] for ex in batch])
        user_history_att_masks = torch.LongTensor([ex['user_history_att_mask'] for ex in batch])

        # print(len(user_histories))
        neg_items = torch.LongTensor([ex['neg_item'] for ex in batch])
        feed_dict = {
            'users': users,
            'user_histories': user_histories,
            'user_history_att_masks': user_history_att_masks,
            'pos_items': pos_items,
            'neg_items': neg_items,
        }
        return feed_dict