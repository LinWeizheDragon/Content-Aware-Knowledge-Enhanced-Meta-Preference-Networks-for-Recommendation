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
from pprint import pprint
from easydict import EasyDict
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from data_loader_manager.data_loader_wrapper import DataLoaderWrapper
from transformers import BertTokenizerFast

class DataLoaderAmazonBooksNRMS(DataLoaderWrapper):
    '''
    Data loader for KBERT training
    Contains 3 datasets for evaluating
    '''
    def __init__(self, config):
        DataLoaderWrapper.__init__(self, config)

        print('Data Loader starts loading data...')
        self.build_dataset()
        self.set_dataloader()

    def set_dataloader(self):
        """This function wraps datasets into dataloader for trainers
        """
        train_dataset_dict = {
            'entity_pairs': self.train_cf_pairs,
            'user_set': self.train_user_set,
            'mode': 'train',
            'n_params': self.n_params,
        }
        self.train_dataset = KBERTDataset(self.config, train_dataset_dict)
        # for i in train_dataset:
        #     print(i)
        train_sampler = RandomSampler(self.train_dataset)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.config.train.batch_size,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=4,
        )
        test_dataset_dict = {
            'entity_pairs': self.test_cf_pairs,
            'user_set': self.test_user_set,
            'mode': 'test',
            'n_params': self.n_params,
        }
        self.test_dataset = KBERTDataset(self.config, test_dataset_dict)

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

    def build_dataset(self):
        """build dataset"""
        self.train_cf, self.test_cf, self.user_dict, n_params, self.graph, self.mat_list = self.load_data()
        self.adj_mat_list, self.norm_mat_list, self.mean_mat_list = self.mat_list

        """cf data"""
        self.train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in self.train_cf], np.int32))
        self.test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in self.test_cf], np.int32))
        print(self.train_cf_pairs.shape, self.test_cf_pairs.shape)
        print('KG Info:', self.n_params)

    def load_data(self):
        '''
        This function loads data from files
        Returns:

        '''
        self.n_params = EasyDict({
            'n_users': int(0),
            'n_items': int(0),
            'n_entities': int(0),
            'n_nodes': int(0),
            'n_relations': int(0)
        })

        self.train_user_set = defaultdict(list)
        self.test_user_set = defaultdict(list)

        directory = os.path.join(self.config.data_loader.additional.train_data_path,
                                 self.config.data_loader.additional.dataset_name)
        train_txt = os.path.join(directory, 'train.txt')
        test_txt = os.path.join(directory, 'test.txt')
        print('reading train and test user-item set ...')
        self.item_interact_count = {}
        train_cf = self.read_cf(train_txt)
        test_cf = self.read_cf(test_txt)
        self.item_interact_count_list = []
        for i in range(1+max(list(self.item_interact_count.keys()))):
            self.item_interact_count_list.append(self.item_interact_count[i])
        self.item_interact_count_list = np.array(self.item_interact_count_list)
        # print('non-empty-ratio:', np.count_nonzero(self.item_interact_count_list)/self.item_interact_count_list.shape[0])
        # input()
        self.load_cbf()
        self.remap_item(train_cf, test_cf)

        print('combinating train_cf and kg data ...')
        kg_final_txt = os.path.join(directory, 'kg_final.txt')
        triplets = self.read_triplets(kg_final_txt)
        self.graph_triplets = triplets
        
        if self.config.train.type == 'KGBERTExecutor':
            # Load graph triplets
            print('building the graph ...')
            graph, relation_dict = self.build_graph(train_cf, triplets)
            print('building the adj mat ...')
            adj_mat_list, norm_mat_list, mean_mat_list = self.build_sparse_relational_graph(relation_dict)
        else:
            # Skip loading graph
            graph, relation_dict = None, None
            adj_mat_list, norm_mat_list, mean_mat_list = None, None, None
        # n_params = easydict({
        #     'n_users': int(n_users),
        #     'n_items': int(n_items),
        #     'n_entities': int(n_entities),
        #     'n_nodes': int(n_nodes),
        #     'n_relations': int(n_relations)
        # })
        user_dict = {
            'train_user_set': self.train_user_set,
            'test_user_set': self.test_user_set
        }
        self.n_params.item_interact_count = self.item_interact_count_list
        return train_cf, test_cf, user_dict, self.n_params, graph, \
               [adj_mat_list, norm_mat_list, mean_mat_list]


    def read_cf(self, file_name):
        inter_mat = list()
        lines = open(file_name, "r").readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(" ")]

            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))
            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])
                self.item_interact_count.setdefault(i_id, 0)
                self.item_interact_count[i_id] += 1

        return np.array(inter_mat)


    def remap_item(self, train_data, test_data):

        self.n_params.n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
        self.n_params.n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1

        for u_id, i_id in train_data:
            self.train_user_set[int(u_id)].append(int(i_id))
        for u_id, i_id in test_data:
            self.test_user_set[int(u_id)].append(int(i_id))


    def read_triplets(self, file_name):

        can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
        # Remove duplications
        can_triplets_np = np.unique(can_triplets_np, axis=0)

        # Here the author increase the relation ids to >1 or >0
        # so that interact == 0 and be interacted == 1 can be accommodated
        if self.config.model_config.inverse_r:
            # get triplets with inverse direction like <entity, is-aspect-of, item>
            inv_triplets_np = can_triplets_np.copy()
            inv_triplets_np[:, 0] = can_triplets_np[:, 2]
            inv_triplets_np[:, 2] = can_triplets_np[:, 0]
            inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
            # consider two additional relations --- 'interact' and 'be interacted'
            can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
            inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
            # get full version of knowledge graph
            triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
        else:
            # consider two additional relations --- 'interact'.
            can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
            triplets = can_triplets_np.copy()

        self.n_params.n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
        self.n_params.n_nodes = self.n_params.n_entities + self.n_params.n_users
        self.n_params.n_relations = max(triplets[:, 1]) + 1

        return triplets


    def build_graph(self, train_data, triplets):
        ckg_graph = nx.MultiDiGraph()
        rd = defaultdict(list)

        print("Begin to load interaction triples ...")
        for u_id, i_id in tqdm(train_data, ascii=True):
            # relation_id 0 refers to interaction between users and items
            rd[0].append([u_id, i_id])

        print("\nBegin to load knowledge graph triples ...")
        for h_id, r_id, t_id in tqdm(triplets, ascii=True):
            # Add KG edges
            ckg_graph.add_edge(h_id, t_id, key=r_id)
            # >1 are other relations
            rd[r_id].append([h_id, t_id])

        return ckg_graph, rd


    def build_sparse_relational_graph(self, relation_dict):
        def _bi_norm_lap(adj):
            # D^{-1/2}AD^{-1/2}
            rowsum = np.array(adj.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        def _si_norm_lap(adj):
            # D^{-1}A
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        adj_mat_list = []
        print("Begin to build sparse relation matrix ...")
        for r_id in tqdm(relation_dict.keys()):
            np_mat = np.array(relation_dict[r_id]) # all items bridged by this relation
            if r_id == 0:
                # interact
                cf = np_mat.copy()
                cf[:, 1] = cf[:, 1] + self.n_params.n_users  # [0, n_items) -> [n_users, n_users+n_items)
                vals = [1.] * len(cf)
                adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(self.n_params.n_nodes, self.n_params.n_nodes))
            else:
                vals = [1.] * len(np_mat)
                adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(self.n_params.n_nodes, self.n_params.n_nodes))
            adj_mat_list.append(adj)

        norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
        mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
        # interaction: user->item, [self.n_params.n_users, self.n_params.n_entities]
        norm_mat_list[0] = norm_mat_list[0].tocsr()[:self.n_params.n_users, self.n_params.n_users:].tocoo()
        mean_mat_list[0] = mean_mat_list[0].tocsr()[:self.n_params.n_users, self.n_params.n_users:].tocoo()

        return adj_mat_list, norm_mat_list, mean_mat_list

    def load_cbf(self):
        """This function loads CBF data for contextual embeddings
        """
        directory = os.path.join(self.config.data_loader.additional.train_data_path,
                                 self.config.data_loader.additional.dataset_name)
        item_list_txt = os.path.join(directory, 'item_list.txt')
        item_description_file = os.path.join(directory, 'book_item_match_with_descriptions.pkl')
        lines = open(item_list_txt, "r").readlines()
        self.asin2id = {}
        self.id2asin = {}
        for l in lines:
            tmps = l.strip()
            row = [i for i in tmps.split(" ")]
            org_id, remap_id, freebase_id = row
            if org_id == 'org_id':
                continue
            self.asin2id[org_id] = int(remap_id)
            self.id2asin[int(remap_id)] = org_id

        print('original asin loaded.')

        with open(item_description_file, 'rb') as f:
            self.book_dict = pickle.load(f)
            # print(len(self.book_dict))
        
        def clean_str(asin, title, input_str):
            try:
                input_str = input_str.strip()
                input_str = input_str.replace('\n', ' ')
                input_str = input_str.replace('\t', ' ')
            except Exception as e:
                print('failling.. ', asin, title, input_str)
                input_str = ''
            return input_str

        # Clean descriptions, removing \n etc.
        for asin, book_item in self.book_dict.items():
            book_item['description'] = clean_str(asin, book_item['title'], book_item['description'])

        print('Total description loaded:', len(self.book_dict))
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.descriptions_in_order = []
        max_id = max(list(self.id2asin.keys()))
        print('tokenizing descriptions...', max_id)
        from utils.text_cleaning import clean_text
        for i in tqdm(range(max_id+1)):
            text = self.book_dict[self.id2asin[i]]['description']
            text = clean_text(text)
            self.descriptions_in_order.append(text)

        print('tokenizing to tensor...', max_id)
        self.tokenized_descriptions = self.tokenizer.batch_encode_plus(
                                        self.descriptions_in_order,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt')
        print(self.tokenized_descriptions['input_ids'].shape)
        # print(self.tokenized_descriptions)
        # print(len(self.tokenized_descriptions_in_order))
        # print(self.item_id2description[0])
        # print(self.tokenized_descriptions_in_order[:2])
        # print(self.tokenizer('this is a test', truncation=True))

        # input()

class KBERTDataset(torch.utils.data.Dataset):
    def __init__(self, config, data):
        self.config = config
        self.entity_pairs = data['entity_pairs']
        self.user_set = data['user_set']
        self.mode = data['mode']
        self.n_params = data['n_params']
        self.item_interact_count = self.n_params.item_interact_count
        if self.config.data_loader.negative_sampling_mode == 'inversed_ratio':
            inverse_count = (1.0/self.item_interact_count)
            self.p =  inverse_count / np.sum(inverse_count)
            print('max prob', np.max(self.p))
            print('min prob', np.min(self.p))
        else:
            self.p = None

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