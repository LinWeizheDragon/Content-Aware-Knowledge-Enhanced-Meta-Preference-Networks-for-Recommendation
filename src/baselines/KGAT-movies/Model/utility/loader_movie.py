import numpy as np
from utility.load_data import Data
from time import time
import scipy.sparse as sp
import random as rd
import collections
import os
import sys
import time
import json
import copy
from tqdm import tqdm
import csv
import json
import pickle
from copy import deepcopy
import numpy as np
import pandas as pd
import random
from time import time
from datetime import datetime
from pprint import pprint
from easydict import EasyDict
from collections import defaultdict

class KGAT_movie_loader(object):
    def __init__(self, args, path):
        self.path = path
        self.args = args

        self.batch_size = args.batch_size
        self.read_movie_user_data()
        print('loading finished.', self.dataset_name)

        # ----------get number of users and items & then load rating data from train_file & test_file------------.
        self.n_train, self.n_test = 0, 0
        self.n_users, self.n_items = 0, 0

        self._load_ratings()
        self.exist_users = self.train_user_dict.keys()

        self._statistic_ratings()

        # ----------get number of entities and relations & then load kg data from kg_file ------------.
        self.n_relations, self.n_entities, self.n_triples = 0, 0, 0
        self.kg_data, self.kg_dict, self.relation_dict = self._load_kg()

        # ----------print the basic info about the dataset-------------.
        self.batch_size_kg = self.n_triples // (self.n_train // self.batch_size)
        # self.batch_size_kg = self.batch_size * 8
        self._print_data_info()
        
        # generate the sparse adjacency matrices for user-item interaction & relational kg data.
        self.adj_list, self.adj_r_list = self._get_relational_adj_list()

        # generate the sparse laplacian matrices.
        self.lap_list = self._get_relational_lap_list()

        # generate the triples dictionary, key is 'head', value is '(tail, relation)'.
        self.all_kg_dict = self._get_all_kg_dict()

        self.all_h_list, self.all_r_list, self.all_t_list, self.all_v_list = self._get_all_kg_data()
        print(len(self.all_h_list), len(self.all_r_list), len(self.all_t_list), len(self.all_v_list))
        print(len(set(self.all_h_list)), len(set(self.all_r_list)), len(set(self.all_t_list)), len(set(self.all_v_list)))
        input()

    # Read movie-user-data
    def read_movie_user_data(self):
        args = self.args
        train_data_path = '/quantus-nfs/users/v-weizhelin/user-movie-data-extended/'
        self.dataset_name = args.movie_dataset_name
        self.preprocessed_movie_structure_data_path = os.path.join(train_data_path, 
                                self.dataset_name, 'preprocessed_movie_structure_data.pkl')
        self.preprocessed_interaction_data_path = os.path.join(train_data_path, 
                                self.dataset_name, 'preprocessed_interaction_data.pkl')
        self.preprocessed_cf_data_path = os.path.join(train_data_path, 
                                self.dataset_name, 'preprocessed_cf_data.pkl')
        
        if os.path.exists(self.preprocessed_movie_structure_data_path):
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
            input('failed to load data!')

        #########################################################
        # Start reading user-item interactions from full data flow
        #########################################################
        if os.path.exists(self.preprocessed_interaction_data_path):
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
            input('Failed to interaction data and KG data')

        

    # reading train & test interaction data.
    # def _load_ratings(self, file_name):
    #     user_dict = dict()
    #     inter_mat = list()

    #     lines = open(file_name, 'r').readlines()
    #     for l in lines:
    #         tmps = l.strip()
    #         inters = [int(i) for i in tmps.split(' ')]

    #         u_id, pos_ids = inters[0], inters[1:]
    #         pos_ids = list(set(pos_ids))

    #         for i_id in pos_ids:
    #             inter_mat.append([u_id, i_id])

    #         if len(pos_ids) > 0:
    #             user_dict[u_id] = pos_ids
    #     return np.array(inter_mat), user_dict

    def _load_ratings(self):
        self.train_user_dict = dict()
        self.train_data = list()
        self.valid_user_dict = dict()
        self.valid_data = list()
        self.test_user_dict = dict()
        self.test_data = list()
        self.cold_start_train_user_dict = dict()
        self.cold_start_train_data = list()
        self.cold_start_test_user_dict = dict()
        self.cold_start_test_data = list()
        
        if os.path.exists(self.preprocessed_cf_data_path):
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
            input('Failed to load preprocessed CF data')
        # TODO: more complicated data split method
        # Now split a users' interactions into 8:1:1
        # print('preparing numpy data')
        # split_train_valid = 0.8
        # split_valid_test = 0.9
        # for user, user_interactions in tqdm(self.filtered_user_interaction_dict.items()):
        #     user_id = self.user2id[user]
        #     # For each user, add interaction data to corresponding cfs
        #     train_valid = int(split_train_valid * len(user_interactions))
        #     valid_test = int(split_valid_test * len(user_interactions))
        #     user_cf = [[user_id,
        #                     self.entity2id[interaction.EntityId]] 
        #                     for interaction in user_interactions]
        #     train_cf += user_cf[:train_valid]
        #     valid_cf += user_cf[train_valid:valid_test]
        #     test_cf += user_cf[valid_test:]

        #     self.train_user_dict[user_id] = [cf[1] for cf in user_cf[:train_valid]]
        #     self.valid_user_dict[user_id] = [cf[1] for cf in user_cf[train_valid:valid_test]]
        #     self.test_user_dict[user_id] = [cf[1] for cf in user_cf[valid_test:]]
        
        for u_id, i_id in train_cf:
            self.train_user_dict.setdefault(int(u_id), []).append(int(i_id))
        for u_id, i_id in valid_cf:
            self.valid_user_dict.setdefault(int(u_id), []).append(int(i_id))
        for u_id, i_id in test_cf:
            self.test_user_dict.setdefault(int(u_id), []).append(int(i_id))
        for u_id, i_id in cold_start_train_cf:
            self.cold_start_train_user_dict.setdefault(int(u_id), []).append(int(i_id))
        for u_id, i_id in cold_start_test_cf:
            self.cold_start_test_user_dict.setdefault(int(u_id), []).append(int(i_id))

        self.train_data = np.array(train_cf)
        self.valid_data = np.array(valid_cf)
        self.test_data = np.array(test_cf)
        self.cold_start_train_data = np.array(cold_start_train_cf)
        self.cold_start_test_data = np.array(cold_start_test_cf)
        

    def _statistic_ratings(self):
        to_build_train_data = np.concatenate([self.train_data, self.cold_start_train_data], axis=0)
        to_build_test_data = np.concatenate([self.test_data, self.cold_start_test_data], axis=0)
        self.n_users = max(max(to_build_train_data[:, 0]), max(to_build_test_data[:, 0])) + 1
        self.n_items = max(max(to_build_train_data[:, 1]), max(to_build_test_data[:, 1])) + 1
        self.n_train = len(self.train_data)
        self.n_test = len(self.test_data)
        print(self.n_users, self.n_items, self.n_train, self.n_test)

    # reading train & test interaction data.
    def _load_kg(self):
        def _construct_kg(kg_np):
            kg = collections.defaultdict(list)
            rd = collections.defaultdict(list)

            for head, relation, tail in kg_np:
                kg[head].append((tail, relation))
                rd[relation].append((head, tail))
            return kg, rd

        kg_np = np.array(self.kg_final)
        kg_np = np.unique(kg_np, axis=0)

        # self.n_relations = len(set(kg_np[:, 1]))
        # self.n_entities = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
        self.n_relations = max(kg_np[:, 1]) + 1
        self.n_entities = max(max(kg_np[:, 0]), max(kg_np[:, 2])) + 1
        self.n_triples = len(kg_np)

        kg_dict, relation_dict = _construct_kg(kg_np)

        return kg_np, kg_dict, relation_dict

    def _print_data_info(self):
        print('[n_users, n_items]=[%d, %d]' % (self.n_users, self.n_items))
        print('[n_train, n_test]=[%d, %d]' % (self.n_train, self.n_test))
        print('[n_entities, n_relations, n_triples]=[%d, %d, %d]' % (self.n_entities, self.n_relations, self.n_triples))
        print('[batch_size, batch_size_kg]=[%d, %d]' % (self.batch_size, self.batch_size_kg))
        

    def _generate_train_cf_batch(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_user_dict[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_i_id = np.random.randint(low=0, high=self.n_items,size=1)[0]

                if neg_i_id not in self.train_user_dict[u] and neg_i_id not in neg_items:
                    neg_items.append(neg_i_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_users_to_test = list(self.test_user_dict.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_user_dict[uid]
            test_iids = self.test_user_dict[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)


        return split_uids, split_state


    
    def _get_relational_adj_list(self):
        t1 = time()
        adj_mat_list = []
        adj_r_list = []

        def _np_mat2sp_adj(np_mat, row_pre, col_pre):
            n_all = self.n_users + self.n_entities
            # single-direction
            a_rows = np_mat[:, 0] + row_pre
            a_cols = np_mat[:, 1] + col_pre
            a_vals = [1.] * len(a_rows)

            b_rows = a_cols
            b_cols = a_rows
            b_vals = [1.] * len(b_rows)

            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
            b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_all, n_all))

            return a_adj, b_adj

        R, R_inv = _np_mat2sp_adj(self.train_data, row_pre=0, col_pre=self.n_users)
        adj_mat_list.append(R)
        adj_r_list.append(0)

        adj_mat_list.append(R_inv)
        adj_r_list.append(self.n_relations + 1)
        print('\tconvert ratings into adj mat done.')

        for r_id in self.relation_dict.keys():
            K, K_inv = _np_mat2sp_adj(np.array(self.relation_dict[r_id]), row_pre=self.n_users, col_pre=self.n_users)
            adj_mat_list.append(K)
            adj_r_list.append(r_id + 1)

            adj_mat_list.append(K_inv)
            adj_r_list.append(r_id + 2 + self.n_relations)
        print('\tconvert %d relational triples into adj mat done. @%.4fs' %(len(adj_mat_list), time()-t1))

        self.n_relations = len(adj_r_list)
        # print('\tadj relation list is', adj_r_list)

        return adj_mat_list, adj_r_list

    def _get_relational_lap_list(self):
        def _bi_norm_lap(adj):
            rowsum = np.array(adj.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        def _si_norm_lap(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.args.adj_type == 'bi':
            lap_list = [_bi_norm_lap(adj) for adj in self.adj_list]
            print('\tgenerate bi-normalized adjacency matrix.')
        else:
            lap_list = [_si_norm_lap(adj) for adj in self.adj_list]
            print('\tgenerate si-normalized adjacency matrix.')
        return lap_list

    def _get_all_kg_dict(self):
        all_kg_dict = collections.defaultdict(list)
        for l_id, lap in tqdm(enumerate(self.lap_list)):

            rows = lap.row
            cols = lap.col

            for i_id in range(len(rows)):
                head = rows[i_id]
                tail = cols[i_id]
                relation = self.adj_r_list[l_id]

                all_kg_dict[head].append((tail, relation))
        return all_kg_dict

    def _get_all_kg_data(self):
        def _reorder_list(org_list, order):
            new_list = np.array(org_list)
            new_list = new_list[order]
            return new_list

        all_h_list, all_t_list, all_r_list = [], [], []
        all_v_list = []

        for l_id, lap in tqdm(enumerate(self.lap_list)):
            all_h_list += list(lap.row)
            all_t_list += list(lap.col)
            all_v_list += list(lap.data)
            all_r_list += [self.adj_r_list[l_id]] * len(lap.row)

        assert len(all_h_list) == sum([len(lap.data) for lap in self.lap_list])

        # resort the all_h/t/r/v_list,
        # ... since tensorflow.sparse.softmax requires indices sorted in the canonical lexicographic order
        print('\treordering indices...')
        org_h_dict = dict()

        for idx, h in enumerate(all_h_list):
            if h not in org_h_dict.keys():
                org_h_dict[h] = [[],[],[]]

            org_h_dict[h][0].append(all_t_list[idx])
            org_h_dict[h][1].append(all_r_list[idx])
            org_h_dict[h][2].append(all_v_list[idx])
        print('\treorganize all kg data done.')

        sorted_h_dict = dict()
        for h in org_h_dict.keys():
            org_t_list, org_r_list, org_v_list = org_h_dict[h]
            sort_t_list = np.array(org_t_list)
            sort_order = np.argsort(sort_t_list)

            sort_t_list = _reorder_list(org_t_list, sort_order)
            sort_r_list = _reorder_list(org_r_list, sort_order)
            sort_v_list = _reorder_list(org_v_list, sort_order)

            sorted_h_dict[h] = [sort_t_list, sort_r_list, sort_v_list]
        print('\tsort meta-data done.')

        od = collections.OrderedDict(sorted(sorted_h_dict.items()))
        new_h_list, new_t_list, new_r_list, new_v_list = [], [], [], []

        for h, vals in od.items():
            new_h_list += [h] * len(vals[0])
            new_t_list += list(vals[0])
            new_r_list += list(vals[1])
            new_v_list += list(vals[2])


        assert sum(new_h_list) == sum(all_h_list)
        assert sum(new_t_list) == sum(all_t_list)
        assert sum(new_r_list) == sum(all_r_list)
        # try:
        #     assert sum(new_v_list) == sum(all_v_list)
        # except Exception:
        #     print(sum(new_v_list), '\n')
        #     print(sum(all_v_list), '\n')
        print('\tsort all data done.')


        return new_h_list, new_r_list, new_t_list, new_v_list

    def _generate_train_A_batch(self):
        exist_heads = self.all_kg_dict.keys()

        if self.batch_size_kg <= len(exist_heads):
            heads = rd.sample(exist_heads, self.batch_size_kg)
        else:
            heads = [rd.choice(exist_heads) for _ in range(self.batch_size_kg)]

        def sample_pos_triples_for_h(h, num):
            pos_triples = self.all_kg_dict[h]
            n_pos_triples = len(pos_triples)

            pos_rs, pos_ts = [], []
            while True:
                if len(pos_rs) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_triples, size=1)[0]

                t = pos_triples[pos_id][0]
                r = pos_triples[pos_id][1]

                if r not in pos_rs and t not in pos_ts:
                    pos_rs.append(r)
                    pos_ts.append(t)
            return pos_rs, pos_ts

        def sample_neg_triples_for_h(h, r, num):
            neg_ts = []
            while True:
                if len(neg_ts) == num: break

                t = np.random.randint(low=0, high=self.n_users + self.n_entities, size=1)[0]
                if (t, r) not in self.all_kg_dict[h] and t not in neg_ts:
                    neg_ts.append(t)
            return neg_ts

        pos_r_batch, pos_t_batch, neg_t_batch = [], [], []

        for h in heads:
            pos_rs, pos_ts = sample_pos_triples_for_h(h, 1)
            pos_r_batch += pos_rs
            pos_t_batch += pos_ts

            neg_ts = sample_neg_triples_for_h(h, pos_rs[0], 1)
            neg_t_batch += neg_ts

        return heads, pos_r_batch, pos_t_batch, neg_t_batch

    def generate_train_batch(self):
        users, pos_items, neg_items = self._generate_train_cf_batch()

        batch_data = {}
        batch_data['users'] = users
        batch_data['pos_items'] = pos_items
        batch_data['neg_items'] = neg_items

        return batch_data

    def generate_train_feed_dict(self, model, batch_data):
        feed_dict = {
            model.users: batch_data['users'],
            model.pos_items: batch_data['pos_items'],
            model.neg_items: batch_data['neg_items'],

            model.mess_dropout: eval(self.args.mess_dropout),
            model.node_dropout: eval(self.args.node_dropout),
        }

        return feed_dict

    def generate_train_A_batch(self):
        heads, relations, pos_tails, neg_tails = self._generate_train_A_batch()

        batch_data = {}

        batch_data['heads'] = heads
        batch_data['relations'] = relations
        batch_data['pos_tails'] = pos_tails
        batch_data['neg_tails'] = neg_tails
        return batch_data

    def generate_train_A_feed_dict(self, model, batch_data):
        feed_dict = {
            model.h: batch_data['heads'],
            model.r: batch_data['relations'],
            model.pos_t: batch_data['pos_tails'],
            model.neg_t: batch_data['neg_tails'],

        }

        return feed_dict


    def generate_test_feed_dict(self, model, user_batch, item_batch, drop_flag=True):

        feed_dict ={
            model.users: user_batch,
            model.pos_items: item_batch,
            model.mess_dropout: [0.] * len(eval(self.args.layer_size)),
            model.node_dropout: [0.] * len(eval(self.args.layer_size)),

        }

        return feed_dict

