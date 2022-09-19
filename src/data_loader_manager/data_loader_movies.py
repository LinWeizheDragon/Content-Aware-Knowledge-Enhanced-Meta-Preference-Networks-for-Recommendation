"""
data_loader_movies.py:  
    Data loader for Collaborative Filtering Models
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

class DataLoaderMovies(DataLoaderWrapper):
    '''
    Data loader for our movie-user dataset
    Compactify the user interactions
    '''

    def __init__(self, config):
        DataLoaderWrapper.__init__(self, config)
        # Load structure data and interaction data from files
        self.load_structure_and_interaction_data()
        # Prepare data for training
        self.build_dataset()
        # Wrap datasets with dataloaders
        self.set_dataloader()
        # input('data loader finished!')

    
    def set_dataloader(self):
        """This function wraps datasets into dataloader for trainers
        """
        train_dataset_dict = {
            'entity_pairs': self.train_cf_pairs,
            'user_set': self.train_user_set,
            'mode': 'train',
            'n_params': self.n_params,
        }
        self.train_dataset = MovieDataset(self.config, train_dataset_dict)
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

        
        # valid_dataset_dict = {
        #     'entity_pairs': self.valid_cf_pairs,
        #     'user_set': self.valid_user_set,
        #     'mode': 'test',
        #     'n_params': self.n_params,
        # }
        # self.valid_dataset = MovieDataset(self.config, valid_dataset_dict)

        # valid_sampler = SequentialSampler(self.valid_dataset)
        # self.valid_dataloader = DataLoader(
        #     self.valid_dataset,
        #     sampler=valid_sampler,
        #     batch_size=self.config.valid.batch_size,
        #     collate_fn=self.valid_dataset.collate_fn,
        #     num_workers=4,
        # )

        test_dataset_dict = {
            'entity_pairs': self.test_cf_pairs,
            'user_set': self.test_user_set,
            'mode': 'test',
            'n_params': self.n_params,
        }
        self.test_dataset = MovieDataset(self.config, test_dataset_dict)

        test_sampler = SequentialSampler(self.test_dataset)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            sampler=test_sampler,
            batch_size=self.config.valid.batch_size,
            collate_fn=self.test_dataset.collate_fn,
            num_workers=4,
        )
        print('statistics: training data loader: {};  test data loader: {}'.format(
                                len(self.train_dataloader), 
                                len(self.test_dataloader)))


    def load_cbf(self):
        """This function loads CBF data for contextual embeddings
        """
        count = 0
        self.item_id2description = {item_id:'' for item_id, item_EntityId in self.id2item.items()}
        for item in tqdm(self.data_dict.values()):
            item_id = self.item2id.get(item['EntityId'], None)
            if item_id: 
                # this item in the dataset
                # Read description to dict
                self.item_id2description[item_id] = item['description']
                count += 1
        print('Total description loaded:', count)
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.descriptions_in_order = []
        max_id = max(list(self.item_id2description.keys()))
        from utils.text_cleaning import clean_text
        print('tokenizing descriptions...', max_id)
        for i in tqdm(range(max_id+1)):
            text = self.item_id2description[i]
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


    def load_structure_and_interaction_data(self):
        """This function reads movie structure data and interaction data
        """
        REGENERATE_STRUCTURE_DATA = self.config.data_loader.additional.regenerate_data
        REGENERATE_INTERACTION_DATA = self.config.data_loader.additional.regenerate_data
        create_dirs([os.path.join(self.config.data_loader.additional.train_data_path, 
                                self.config.data_loader.dataset_name)])
        self.preprocessed_movie_structure_data_path = os.path.join(self.config.data_loader.additional.train_data_path, 
                                self.config.data_loader.dataset_name, 'preprocessed_movie_structure_data.pkl')
        self.preprocessed_interaction_data_path = os.path.join(self.config.data_loader.additional.train_data_path, 
                                self.config.data_loader.dataset_name, 'preprocessed_interaction_data.pkl')
        
        
        #########################################################
        # Start reading movie structure data
        #########################################################
        if os.path.exists(self.preprocessed_movie_structure_data_path) and REGENERATE_STRUCTURE_DATA == False:
            # Read data instead of re-generate
            print('reading preprocessed data from', self.preprocessed_movie_structure_data_path)
            with open(self.preprocessed_movie_structure_data_path, "rb" ) as f:
                load_pickle_data = pickle.load(f)
                self.entity2id = load_pickle_data['entity2id']
                self.item2id = load_pickle_data['item2id']
                # all_nodes = load_pickle_data['all_nodes']
                all_nodes_lookup = load_pickle_data['all_nodes_lookup']
                all_movie_lookup = load_pickle_data['all_movie_lookup']
                all_entity_lookup = load_pickle_data['all_entity_lookup']
                self.data_dict = load_pickle_data['data_dict']

                self.id2item = {v:k for k, v in self.item2id.items()}
                self.id2entity = {v:k for k, v in self.entity2id.items()}
            
            print('All nodes (movies, entities) in the graph:', len(all_nodes_lookup.keys()))
            print('Movie nodes:', len(all_movie_lookup.keys()))
            print('all KG entities (without users):', len(all_entity_lookup.keys()))
            print('item2id mapping:', len(self.item2id))
            print('entity2id mapping:', len(self.entity2id))


            
        else:
            tsv_path = os.path.join(self.config.data_loader.additional.train_data_path, 
                                    'movie_structure_data.tsv')
            print('start reading data from', tsv_path)

            self.data_df = pd.read_csv(tsv_path, sep='\t', header=0, error_bad_lines=False)
            self.data_df = self.data_df.replace(['nan', 'None', 'NaN', np.nan], '')
            for col in self.data_df.columns:
                self.data_df[col] = self.data_df[col].str.strip()
            new_cols = [col.strip() for col in self.data_df.columns]
            self.data_df.columns = new_cols
            TOTAL_MOVIE_NUM = self.config.data_loader.additional.load_num_movies
            frac = TOTAL_MOVIE_NUM / self.data_df.shape[0]
            print('sampling fraction:', frac)
            # self.data_df = self.data_df.sample(frac=frac)
            self.data_dict = self.data_df.to_dict(orient='index')


            ###### Use pre-processed frequent movie list #######
            print('reading pre-processed movie list!')
            pickle_path = os.path.join(self.config.data_loader.additional.train_data_path,
                                     'movie_user_count.pkl')
            with open(pickle_path, "rb" ) as f:
                load_pickle_data = pickle.load(f)
                sorted_movie_user_count = load_pickle_data['sorted_movie_user_count']
            movie_user_count_dict = {}
            
            DROP_MOVIE_NUM = 50
            print('dropped first {} movies'.format(DROP_MOVIE_NUM))
            # Drop some top movies
            for item in sorted_movie_user_count[DROP_MOVIE_NUM:TOTAL_MOVIE_NUM+DROP_MOVIE_NUM]:
                movie_user_count_dict[item[0]] = item[1]

            self.data_dict = {k:v for k, v in tqdm(self.data_dict.items()) if v['EntityId'] in movie_user_count_dict.keys()}
            print('selected movies:', len(self.data_dict.keys()))

            MULTIPLE_TERMS = ['actor', 'country', 'director', 'editor',
                            'genre', 'gross_revenue', 'honors', 'language',
                            'minutes', 'name', 'producer', 'production_company', 'rating',
                            'writer']
            
            # Split columns where multiple values are concatenated by |||
            for item in tqdm(self.data_dict.values()):
                for col, value in item.items():
                    if col in MULTIPLE_TERMS:
                        # print(col, value)
                        if value == np.nan:
                            value = []
                        elif value == '':
                            value = []
                        else:
                            value = value.split('|||')

                        item[col] = value
            
            ## Print a sample
            # for i in self.data_dict.values():
            #     pprint(i)
            #     break

            ## Start preprocessing data
            all_nodes = []
            all_nodes_lookup = {}
            
            # These terms contain nodes
            CONTAIN_NODE_TERMS = ['actor', 'country', 'director', 'editor',
                            'genre', 'honors', 'language',
                            'producer', 'production_company', 'rating',
                            'writer']
            
            for item in tqdm(self.data_dict.values()):
                # if len(all_nodes_lookup.keys()) > TOTAL_MOVIE_NUM:
                #     # Reduce dataset size
                #     break
                for col, values in item.items():
                    # Each record represents a movie-user interaction
                    if col == 'EntityId':
                        # add to nodes
                        node_dict = {
                            'name': values,
                            'type': 'movie',
                            'name_list': item['name'],
                        }
                        all_nodes_lookup.setdefault(values, node_dict)

                    # Extract node info from this record
                    if col in CONTAIN_NODE_TERMS:
                        for v in values:
                            node_dict = {
                                'name': v,
                                'type': col,
                            }
                            all_nodes_lookup.setdefault(v, node_dict)

            
            print('All nodes (movies, entities) in the graph:', len(all_nodes_lookup.keys()))

            all_movie_lookup = {node_name:node_dict for node_name,node_dict in all_nodes_lookup.items() if node_dict['type'] == 'movie'}
            all_entity_lookup = {node_name:node_dict for node_name,node_dict in all_nodes_lookup.items() if node_dict['type'] != 'user'}

            print('Movie nodes:', len(all_movie_lookup.keys()))
            print('all KG entities (without users):', len(all_entity_lookup.keys()))

            # Create movie item mapping
            self.item2id = {}
            self.item2id = {k:id for id, k in enumerate(all_movie_lookup.keys())}
            self.id2item = {v:k for k, v in self.item2id.items()}
            print('item2id mapping:', len(self.item2id))

            # Create entity mapping
            # incorporates items
            self.entity2id = self.item2id.copy()
            for k, v in all_entity_lookup.items():
                if v['type'] not in ['movie', 'user']:
                    # append to the back of movie items and create ids
                    self.entity2id[k] = len(self.entity2id)
            print('entity2id mapping:', len(self.entity2id))

            
            pickle_to_save = {
                'entity2id': self.entity2id,
                'item2id': self.item2id,
                # 'all_nodes': all_nodes,
                'all_movie_lookup': all_movie_lookup,
                'all_entity_lookup': all_entity_lookup,
                'all_nodes_lookup': all_nodes_lookup,
                'data_dict': self.data_dict,
            }
            with open(self.preprocessed_movie_structure_data_path, "wb" ) as f:
                print('saving preprocessed data...')
                pickle.dump(pickle_to_save, f)
                print('preprocessed data has been saved to', self.preprocessed_movie_structure_data_path)

        # Read descriptions from movie structure data
        # Load description for CBF
        self.load_cbf()

        #########################################################
        # Start reading user-item interactions from full data flow
        #########################################################
        if os.path.exists(self.preprocessed_interaction_data_path) and REGENERATE_INTERACTION_DATA == False:
            # Read data instead of re-generate
            print('reading preprocessed data from', self.preprocessed_interaction_data_path)
            with open(self.preprocessed_interaction_data_path, "rb" ) as f:
                load_pickle_data = pickle.load(f)
                self.filtered_user_interaction_dict = load_pickle_data['filtered_user_interaction_dict']
                self.id2relation = load_pickle_data['id2relation']
                self.kg_final = load_pickle_data['kg_final']
                self.relation2id = {v:k for k, v in self.id2relation.items()}
            print('after filtering, there are users:', len(self.filtered_user_interaction_dict))
            
            count_interactions = []
            all_users_lookup = {}
            for user, user_interactions in tqdm(self.filtered_user_interaction_dict.items()):
                count_interactions.append(len(user_interactions))
                # Now, add users to all nodes
                node_dict = {
                    'name': user,
                    'type': 'user',
                }
                all_nodes_lookup[user] = node_dict # add to nodes
                all_users_lookup[user] = node_dict # add to users

            print('avg user interaction number:', 
                    np.mean(np.array(count_interactions)))
                
            # Create user mapping
            self.user2id = {}
            self.user2id = {k:id for id, k in enumerate(all_users_lookup.keys())}
            self.id2user = {v:k for k, v in self.user2id.items()}
            print('user2id mapping:', len(self.user2id))
            print('User nodes:', len(all_users_lookup.keys()))
            print('All nodes (users, movies, entities) in the graph:', len(all_nodes_lookup.keys()))
            
            print('final KG triplets:', len(self.kg_final))
        else:
            self.user_interaction_dict = {}
            not_matched_items = []

            for tsv_index, tsv_file_name in enumerate(self.config.data_loader.additional.file_list):
                print('reading {} / {} tsv files...'.format(tsv_index+1, len(self.config.data_loader.additional.file_list)))
                tsv_path = os.path.join(self.config.data_loader.additional.train_data_path, 
                                        tsv_file_name)
                print('reading user interaction data from', tsv_path)
                with open(tsv_path) as csv_file:
                    count_total = sum(1 for row in csv_file)
                    csv_file.seek(0)
                    csv_reader = csv.reader(csv_file, delimiter='\t')
                    next(csv_reader) # Skip the header
                    for row in tqdm(csv_reader, total=count_total):
                        UserId, EntityId, RequestTime, DwellTime = row
                        interaction = EasyDict()
                        interaction.UserId = UserId
                        interaction.EntityId = EntityId
                        interaction.RequestTime = RequestTime
                        interaction.DwellTime = int(DwellTime)
                        
                        if interaction.EntityId not in self.entity2id.keys():
                            not_matched_items.append(interaction.EntityId)
                        else:
                            if interaction.DwellTime > 10:
                                # add to user interactions
                                self.user_interaction_dict.setdefault(UserId, []).append(interaction)
                            else:
                                pass
                # break # TODO Debug use

            not_matched_items = list(set(not_matched_items))
            print('There are {} movies that are not in the structure data'.format(len(not_matched_items)))
            
            # Compatify the user interactions
            print('compactify the user interactions....')
            self.filtered_user_interaction_dict = {}
            for user_id, user_interactions in tqdm(self.user_interaction_dict.items()):
                last_entity_id = None
                for interaction in user_interactions:
                    if last_entity_id == interaction.EntityId:
                        self.filtered_user_interaction_dict.setdefault(user_id, [])[-1].DwellTime += interaction.DwellTime
                    else:
                        # add to filtered list
                        self.filtered_user_interaction_dict.setdefault(user_id, []).append(interaction)
                    last_entity_id = interaction.EntityId
            
            # Sort interactions by time
            # print('sorting interactions by time')
            # for user, user_interactions in tqdm(self.filtered_user_interaction_dict.items()):
            #     datetime.strptime(date_string, '')

            # Count user statistics and filter non-active users
            self.filtered_user_interaction_dict = {user: user_interactions 
                                                    for user, user_interactions in tqdm(self.filtered_user_interaction_dict.items())
                                                    if len(user_interactions) >= 10}
            print('after interaction number filtering, there are users:', len(self.filtered_user_interaction_dict))
            
            # Sort users by number of interactions
            sorted_user_interaction_count = sorted(self.filtered_user_interaction_dict.items(),key=lambda item:len(item[1]), reverse=True)
            
            if self.config.data_loader.additional.load_num_users > 0:
                # If num_users is set by arguments  
                load_users = min(self.config.data_loader.additional.load_num_users, 
                                len(sorted_user_interaction_count))
            else:
                # Else take all users for training
                load_users = len(sorted_user_interaction_count)
            
            # Get most-active users
            self.filtered_user_interaction_dict = {item[0]:item[1] for item in sorted_user_interaction_count[:load_users]}
            print('load number of users:', len(self.filtered_user_interaction_dict.keys()))
            
            count_interactions = []
            all_users_lookup = {}
            for user, user_interactions in tqdm(self.filtered_user_interaction_dict.items()):
                count_interactions.append(len(user_interactions))
                # Now, add users to all nodes
                node_dict = {
                    'name': user,
                    'type': 'user',
                }
                all_nodes_lookup[user] = node_dict # add to nodes
                all_users_lookup[user] = node_dict # add to users

            print('avg user interaction number:', 
                    np.mean(np.array(count_interactions)))
                

            # Create user mapping
            self.user2id = {}
            self.user2id = {k:id for id, k in enumerate(all_users_lookup.keys())}
            self.id2user = {v:k for k, v in self.user2id.items()}
            print('user2id mapping:', len(self.user2id))
            print('User nodes:', len(all_users_lookup.keys()))
            print('All nodes (users, movies, entities) in the graph:', len(all_nodes_lookup.keys()))
            
            #########################################################
            # Start processing final KG
            #########################################################

            # Create relation lookups
            relation_list = ['actor', 'country', 'director', 'editor',
                                'genre', 'honors', 'language',
                                'minutes', 'producer', 'production_company', 'rating',
                                'writer']
            self.id2relation = {}
            for r_id, relation in enumerate(relation_list):
                self.id2relation[r_id] = relation
            self.relation2id = {v:k for k, v in self.id2relation.items()}

            # Create KG entity-relation-entity triplets
            processed_entity_id = []
            self.kg_final = []
            for item in tqdm(self.data_dict.values()):
                if item['EntityId'] in processed_entity_id:
                    # print('skipped', item['EntityId'])
                    continue
                for col, values in item.items():
                    if col in relation_list:
                        for v in values:
                            try:
                                # Extract KG triplet
                                h_id = self.entity2id[item['EntityId']]
                                r_id = self.relation2id[col]
                                t_id = self.entity2id[v]
                                triplet = (h_id, r_id, t_id)
                                # print('{}-->>{}-->>{}'.format(item['EntityId'],
                                #                         col, v))
                                # print('{}-->>{}-->>{}'.format(h_id, r_id, t_id))
                                self.kg_final.append(triplet)
                            except Exception as e:
                                # do not add this kg
                                pass
                
                processed_entity_id.append(item['EntityId'])        
            print('final KG triplets:', len(self.kg_final))
            
            pickle_to_save = {
                'filtered_user_interaction_dict':self.filtered_user_interaction_dict,
                'kg_final': self.kg_final,
                'id2relation': self.id2relation,
            }
            with open(self.preprocessed_interaction_data_path, "wb" ) as f:
                print('saving preprocessed data...')
                pickle.dump(pickle_to_save, f)
                print('preprocessed interaction data has been saved to', self.preprocessed_interaction_data_path)
        

    def build_dataset(self):
        """build dataset"""
        self.train_cf, self.valid_cf, self.test_cf, self.cold_start_train_cf, self.cold_start_test_cf, self.user_dict, n_params, self.graph, self.mat_list = self.load_data()
        self.adj_mat_list, self.norm_mat_list, self.mean_mat_list = self.mat_list

        """cf data"""
        self.train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in self.train_cf], np.int32))
        self.test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in self.test_cf], np.int32))
        print(self.train_cf_pairs.shape, self.test_cf_pairs.shape)
        print('KG Info:', self.n_params)

    def load_data(self):
        """This function prepares data for training

        Returns:
            [type]: [description]
        """

        REGENERATE_CF_DATA = self.config.data_loader.additional.regenerate_data


        self.n_params = EasyDict({
            'n_users': len(self.user2id.keys()),
            'n_items': len(self.item2id.keys()),
            'n_entities': len(self.entity2id.keys()),
            'n_nodes': int(0),
            'n_relations': int(0),
        })

        self.train_user_set = defaultdict(list)
        self.valid_user_set = defaultdict(list)
        self.test_user_set = defaultdict(list)
        self.cold_start_train_user_set = defaultdict(list)
        self.cold_start_test_user_set = defaultdict(list)
        
        # configs of splitting cold-start users
        split_cold_start = self.config.data_loader.additional.split_cold_start
        cold_start_ratio = self.config.data_loader.additional.cold_start_ratio
        
        self.preprocessed_cf_data_path = os.path.join(self.config.data_loader.additional.train_data_path, 
                                self.config.data_loader.dataset_name, 'preprocessed_cf_data.pkl')
        
        if os.path.exists(self.preprocessed_cf_data_path) and REGENERATE_CF_DATA == False:
            # reading preprocessed cf data
            print('reading preprocessed data from', self.preprocessed_cf_data_path)
            with open(self.preprocessed_cf_data_path, "rb" ) as f:
                load_pickle_data = pickle.load(f)
                train_cf = load_pickle_data['train_cf']
                valid_cf = load_pickle_data['valid_cf']
                test_cf = load_pickle_data['test_cf']
                cold_start_train_cf = load_pickle_data.get('cold_start_train_cf', np.array([]))
                cold_start_test_cf = load_pickle_data.get('cold_start_test_cf', np.array([]))
                
        else:
            train_cf = []
            valid_cf = []
            test_cf = []
            cold_start_train_cf = []
            cold_start_test_cf = []

            # TODO: more complicated data split method
            # Now split a users' interactions into 8:1:1
            print('preparing numpy data')
            split_train_valid = 0.8
            split_valid_test = 0.9
            if split_cold_start:
                # Split a proportion of users into cold-start test set
                num_cold_start_users = int(cold_start_ratio *  len(self.filtered_user_interaction_dict))
                num_normal_users = len(self.filtered_user_interaction_dict) - num_cold_start_users
                print('trying to split {} users for cold-start testing. {} users as normal.'.format(num_cold_start_users, num_normal_users))
            
            
            processed_user_num = 0
            for user, user_interactions in tqdm(self.filtered_user_interaction_dict.items()):
                # For each user, add interaction data to corresponding cfs
                train_valid = int(split_train_valid * len(user_interactions))
                valid_test = int(split_valid_test * len(user_interactions))
                user_cf = [[self.user2id[interaction.UserId],
                                self.entity2id[interaction.EntityId]] 
                                for interaction in user_interactions]
                if split_cold_start and processed_user_num >= num_normal_users:
                    # Add all interactions of cold-start users
                    cold_start_train_cf += user_cf[:train_valid]
                    cold_start_test_cf += user_cf[train_valid:]
                else:
                    # split into train/valid/test
                    train_cf += user_cf[:train_valid]
                    valid_cf += user_cf[train_valid:valid_test]
                    test_cf += user_cf[valid_test:]

                processed_user_num += 1

            # Transform into numpy array
            train_cf = np.array(train_cf)
            valid_cf = np.array(valid_cf)
            test_cf = np.array(test_cf)
            cold_start_train_cf = np.array(cold_start_train_cf)
            cold_start_test_cf = np.array(cold_start_test_cf)

            pickle_to_save = {
                'train_cf':train_cf,
                'valid_cf':valid_cf,
                'test_cf':test_cf,
                'cold_start_train_cf': cold_start_train_cf,
                'cold_start_test_cf': cold_start_test_cf,
            }
            with open(self.preprocessed_cf_data_path, "wb" ) as f:
                print('saving preprocessed data...')
                pickle.dump(pickle_to_save, f)
                print('preprocessed interaction data has been saved to', self.preprocessed_cf_data_path)
        
        
        print('train_cf', train_cf.shape)
        print('valid_cf', valid_cf.shape)
        print('test_cf', test_cf.shape)
        print('cold_start_train_cf', cold_start_train_cf.shape)
        print('cold_start_test_cf', cold_start_test_cf.shape)

        self.item_interact_count = {}
        for cf_data in [train_cf, valid_cf, test_cf, cold_start_train_cf, cold_start_test_cf]:
            cf_data_list = cf_data.tolist()
            for cf_row in tqdm(cf_data_list):
                u_id, i_id = cf_row
                self.item_interact_count.setdefault(i_id, 0)
                self.item_interact_count[i_id] += 1
        
        self.item_interact_count_list = []
        for i in range(len(self.item2id.keys())):
            count = self.item_interact_count.get(i, 0)
            self.item_interact_count_list.append(count)
        self.item_interact_count_list = np.array(self.item_interact_count_list)
        print(self.item_interact_count_list)
        print(self.item_interact_count_list.shape)
        
        # Remap and get info
        print('remapping data...')
        # self.remap_item(train_cf, test_cf)
        
        for u_id, i_id in train_cf:
            self.train_user_set[int(u_id)].append(int(i_id))
        for u_id, i_id in valid_cf:
            self.valid_user_set[int(u_id)].append(int(i_id))
        for u_id, i_id in test_cf:
            self.test_user_set[int(u_id)].append(int(i_id))
        for u_id, i_id in cold_start_train_cf:
            self.cold_start_train_user_set[int(u_id)].append(int(i_id))
        for u_id, i_id in cold_start_test_cf:
            self.cold_start_test_user_set[int(u_id)].append(int(i_id))

        print('combinating train_cf and kg data ...')
        triplets = self.read_triplets()
        self.graph_triplets = triplets
        
        if self.config.train.type != 'KBERTExecutor':
            print('building the graph ...')

            # Keep the relation dict unchanged
            if split_cold_start:
                to_build_cf = np.concatenate([train_cf, cold_start_train_cf], axis=0)
            else:
                to_build_cf = train_cf

            graph, relation_dict = self.build_graph(to_build_cf, triplets)

            print('building the adj mat ...')
            adj_mat_list, norm_mat_list, mean_mat_list = self.build_sparse_relational_graph(relation_dict)
        
        else:

            # Skip loading graph
            graph, relation_dict = None, None
            adj_mat_list, norm_mat_list, mean_mat_list = None, None, None

        user_dict = {
            'train_user_set': self.train_user_set,
            'valid_user_set': self.valid_user_set,
            'test_user_set': self.test_user_set,
            'cold_start_test_user_set': self.cold_start_test_user_set,
            'cold_start_train_user_set': self.cold_start_train_user_set,
        }
        self.n_params.item_interact_count = self.item_interact_count_list
        return train_cf, valid_cf, test_cf, cold_start_train_cf, cold_start_test_cf, user_dict, self.n_params, graph, \
               [adj_mat_list, norm_mat_list, mean_mat_list]


    # def remap_item(self, train_data, test_data):

    #     self.n_params.n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
    #     self.n_params.n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1

    #     for u_id, i_id in train_data:
    #         self.train_user_set[int(u_id)].append(int(i_id))
    #     for u_id, i_id in test_data:
    #         self.test_user_set[int(u_id)].append(int(i_id))

    def read_triplets(self):
        can_triplets_np = np.array(self.kg_final)
        print(can_triplets_np.shape)
        # can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
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
        # edge_tuples = []
        for h_id, r_id, t_id in tqdm(triplets, ascii=True):
            # Add KG edges
            # edge_tuples.append((h_id, t_id, r_id, {}))
            ckg_graph.add_edge(h_id, t_id, key=r_id)
            # >1 are other relations
            rd[r_id].append([h_id, t_id])
        # print('adding edges to graph')
        # ckg_graph.add_edges_from(edge_tuples)
        # print('adding edges completed.')

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




class MovieDataset(torch.utils.data.Dataset):
    def __init__(self, config, data):
        self.config = config
        self.entity_pairs = data['entity_pairs']
        self.user_set = data['user_set']
        self.mode = data['mode']
        self.n_params = data['n_params']
        self.item_interact_count = self.n_params.item_interact_count
        if self.config.data_loader.negative_sampling_mode == 'inversed_ratio':
            inverse_count = (1.0/self.item_interact_count)
            inverse_count[np.isinf(inverse_count)] = 0
            self.p =  inverse_count / np.sum(inverse_count)
            print('max prob', np.max(self.p))
            print('min prob', np.min(self.p))
        else:
            self.p = None

    def __len__(self):
        return len(self.entity_pairs)

    def __getitem__(self, idx):
        def negative_sampling(user_item, num_samples=1):
            """Generate negative samples for a user. ONLY used in training

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
        
        feed_dict = {
            'user': user,
            'pos_item': pos_item,
        }
        if self.mode == 'train':
            neg_items = negative_sampling(user)
            # print(user, pos_item, neg_items)
            # currently using only the first item
            neg_item = neg_items[0]
            feed_dict['neg_item'] = neg_item

        return feed_dict

    
    def collate_fn(self, batch):  # optional but useful
        '''
            when collate_fn is given to the torch dataloader, we can do further actions to the batch, e.g., tensor can be formed here
            a batch is formed as a list where each element is a defined data returned by __getitem__, andy
        '''
        users = torch.LongTensor([ex['user'] for ex in batch])
        pos_items = torch.LongTensor([ex['pos_item'] for ex in batch])
        if self.mode == 'train':
            neg_items = torch.LongTensor([ex['neg_item'] for ex in batch])
        feed_dict = {
            'users': users,
            'pos_items': pos_items,
            'neg_items': neg_items if self.mode == 'train' else None,
        }
        return feed_dict