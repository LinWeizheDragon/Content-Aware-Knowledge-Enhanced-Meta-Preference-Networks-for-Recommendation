"""
KMPN_executor.py:  Training functions for KMPN/CKMPN
"""

__author__ = "Weizhe Lin"
__copyright__ = "Copyright 2021, Weizhe Lin"
__version__ = "1.0.0"
__email__ = "wl356@cam.ac.uk"
__status__ = "Published for Github"

import math
import time
import os
import sys
import scipy
import datetime
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pickle

from train_executor.base_executor import BaseExecutor
from utils.log_system import logger
from utils.dirs import *
from tqdm import tqdm
from easydict import EasyDict
import multiprocessing
from torch.multiprocessing import Manager, spawn, Process
import heapq
from functools import partial
from prettytable import PrettyTable
from pprint import pprint
import hashlib

from utils.wl_test import WLTestKernel
from utils.KGIN_evaluate import ranklist_by_heapq, get_auc, ranklist_by_sorted, get_performance
from transformers import get_linear_schedule_with_warmup
from models.KMPN.KMPN import Recommender

class KMPNExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        BaseExecutor.__init__(self, config, data_loader)
        logger.print('Training has been initiated.')
        
        # default to None
        bert_emb_data = None
        if 'NRMS_BERT_EMB' in self.config.model_config.module_type:
            
            from utils.dirs import create_dirs
            create_dirs([os.path.join(self.config.data_loader.additional.train_data_path, 
                                self.config.data_loader.dataset_name)])
            preprocessed_bert_emb_data_path = os.path.join(self.config.data_loader.additional.train_data_path, 
                                self.config.data_loader.dataset_name, 'preprocessed_NRMS_emb_data.pt')
            if os.path.exists(preprocessed_bert_emb_data_path) and not self.config.data_loader.additional.regenerate_transformer_data:
                # Load from file
                print('Loading preprocessed NRMS_BERT embedding data from', preprocessed_bert_emb_data_path)
                bert_emb_data = torch.load(preprocessed_bert_emb_data_path)
                print('preprocessed NRMS_BERT embedding data loaded from', preprocessed_bert_emb_data_path)
            else:
                from models.NRMS_BERT.NRMS_BERT_v2 import NRMS
                bert_emb_data = {}
                # Init NRMS model
                tokenized_descriptions = data_loader.tokenized_descriptions.copy()
                self.NRMS = NRMS(config, tokenized_descriptions)
                # Try to load pretrained NRMS weights
                load_transformer_path = self.config.model_config.load_transformer_path
                print('loading NRMS_BERT weights from', load_transformer_path)
                checkpoint = torch.load(load_transformer_path)
                pretrained_dict = checkpoint['state_dict']
                model_dict = self.NRMS.state_dict()
                model_dict.update(pretrained_dict)
                self.NRMS.load_state_dict(model_dict)
                print('loaded NRMS_BERT weights from epoch', checkpoint['epoch'])
                print(self.NRMS)
                if self.config.using_data_parallel:
                    # Using Mutiple GPUs
                    self.NRMS = nn.DataParallel(self.NRMS)
                self.NRMS.to(self.config.device)
                print('NRMS_BERT model loaded.')
                
                # generate item pre-trained embeddings
                with torch.no_grad():
                    self.NRMS.eval()
                    all_item_embed, all_user_embed = self.generate_NRMS_embeddings()

                print(all_item_embed.shape, all_user_embed.shape)

                bert_emb_data['item_bert_emb'] = all_item_embed
                bert_emb_data['user_bert_emb'] = all_user_embed
                # Release model from GPUs
                self.NRMS.cpu()
                del self.NRMS
                # save to file
                torch.save(bert_emb_data, preprocessed_bert_emb_data_path)
        
        
        
        """define model"""
        self.model = Recommender(
            config,
            data_loader.n_params,
            data_loader.graph,
            data_loader.mean_mat_list[0],
            bert_emb_data=bert_emb_data).to(config.device)

        if 'NRMS_BERT_EMB' in self.config.model_config.module_type and self.config.model_config.load_graph_path != '':
            load_graph_path = self.config.model_config.load_graph_path
            print('loading KMPN weights from {}'.format(load_graph_path))
            checkpoint = torch.load(load_graph_path)
            pretrained_dict = checkpoint['state_dict']
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            print(pretrained_dict.keys())
            model_dict.update(pretrained_dict)
            print('KMPN weights loaded from epoch', checkpoint['epoch'])
            self.model.load_state_dict(model_dict)

        if self.config.using_data_parallel:
            # Using Mutiple GPUs
            self.model = nn.DataParallel(self.model)

        self.model.to(self.config.device)
        

        print(self.model)

        """define optimizer"""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.train.lr)

        """The below content is not presented in the paper"""
        '''
        if 'NRMS_BERT_EMB_ITER_TRAIN' in self.config.model_config.module_type:
            del self.optimizer
            graph_linear_parameters = filter(lambda p: 'dual' not in p[0] or 'merge' in p[0], self.model.named_parameters())
            transformer_linear_parameters = filter(lambda p: 'dual' in p[0] or 'merge' in p[0], self.model.named_parameters())
            print('===graph parameters===')
            print([name for name, param in graph_linear_parameters])
            print('===transformer parameters===')
            print([name for name, param in transformer_linear_parameters])
            self.graph_optimizer = optim.Adam(
                [
                    dict(params=[param for name, param in graph_linear_parameters],
                    lr=self.config.train.graph_lr),
                ]
            )
            self.transformer_optimizer = optim.Adam(
                [
                    dict(params=[param for name, param in transformer_linear_parameters],
                    lr=self.config.train.lr),
                ]
            )
            # default opt
            self.optimizer = self.transformer_optimizer
        
        if 'NRMS_BERT_EMB_FREEZE_KGIN' in self.config.model_config.module_type:
            for name, param in self.model.named_parameters():
                if not 'dual' in name and not 'merge' in name:
                    param.requires_grad = False
                # if 'bert.encoder.layer' in name:
                #     layer_num = int(name.split('bert.encoder.layer.')[1].split('.')[0])
                #     if layer_num < self.config.model_config.freeze_transformer_layers:
                #         param.requires_grad = False
                # if 'bert.embeddings' in name:
                #     param.requires_grad = False
                print(name, param.requires_grad)

            transformer_parameters = filter(lambda p: p[1].requires_grad, self.model.named_parameters())
            self.optimizer = optim.Adam(
                [
                    dict(params=[param for name, param in transformer_parameters],
                    lr=self.config.train.lr),
                ]
            )

        if 'NRMS_BERT_EMB_FREEZE_ALL' in self.config.model_config.module_type:
            for name, param in self.model.named_parameters():
                if not 'merge' in name:
                    param.requires_grad = False
                print(name, param.requires_grad)
            linear_parameters = filter(lambda p: p[1].requires_grad, self.model.named_parameters())
            self.optimizer = optim.Adam(
                [
                    dict(params=[param for name, param in linear_parameters],
                    lr=self.config.train.lr),
                ]
            )
        '''

        self.train_data_loader = self.data_loader.train_dataloader
        self.test_data_loader = self.data_loader.test_dataloader


        if self.config.train.scheduler == 'linear':
            # Using Linear scheduler
            # Calculate total iterations to execute, apply linear schedule
            t_total = len(
                self.train_data_loader) // self.config.train.additional.gradient_accumulation_steps * (
                                self.config.train.epochs)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.train.additional.warmup_steps,
                num_training_steps=t_total
            )
        elif self.config.train.scheduler == 'cosine':
            t_total = self.config.train.epochs
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                            t_total, eta_min=1e-5, last_epoch=-1, verbose=False)
        else:
            self.scheduler = None

        # Load checkpoints
        self.load_checkpoint_model(load_epoch=self.config.train.load_epoch,
                                   load_best_model=self.config.train.load_best_model,
                                   load_model_path=self.config.train.load_model_path)

        logger.print("finished initialization...starting training.")

    def run_test(self):
        """This function is used for "--mode test"
        """
        test_s_t = time.time()
        with torch.no_grad():
            self.model.eval()
            ret = self.test(save=True)
        test_e_t = time.time()
        train_res = PrettyTable()
        train_res.field_names = ["Epoch", "tesing time", "recall", "ndcg", "precision", "hit_ratio"]
        train_res.add_row(
            [self.loaded_epoch, test_e_t - test_s_t, ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
        )
        print(train_res)
        input('testing done!')

    def train(self):
        logger.print('Training started.')
        #############################################
        #
        #                load setup
        #
        #############################################
        batch_size = self.config.train.batch_size
        save_interval = self.config.train.save_interval
        device = self.config.device
        gradient_accumulation_steps = self.config.train.additional.gradient_accumulation_steps
        start_time = datetime.datetime.now()

        
        logdir = os.path.join(self.config.tensorboard_path)  # datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        if self.config.ITP_tensorboard_path:
            # redirect to ITP TB path
            logdir = self.config.ITP_tensorboard_path

        logger.print(logdir)
        # For debug use
        # logdir = self.config.experiment_name

        ADDED_GRAPH = True
        if self.config.reset:
            ADDED_GRAPH = False

        if self.loaded_epoch == 0:
            ADDED_GRAPH = False

        writer = SummaryWriter(logdir)


        # Here, we save best performance across all epochs for future evaluation
        best_performance = {}
        init_dict = {'best': 0, 'epoch': -1, 'valid_best': 0, 'valid_epoch': -1}
        for index, K in enumerate(self.config.model_config.Ks):
            best_performance.setdefault('recall_at_{}'.format(K), init_dict)
            best_performance.setdefault('ndcg_at_{}'.format(K), init_dict)
            best_performance.setdefault('precision_at_{}'.format(K), init_dict)
            best_performance.setdefault('hit_ratio_at_{}'.format(K), init_dict)
            best_performance.setdefault('cold_start_recall_at_{}'.format(K), init_dict)
            best_performance.setdefault('cold_start_ndcg_at_{}'.format(K), init_dict)
            best_performance.setdefault('cold_start_precision_at_{}'.format(K), init_dict)
            best_performance.setdefault('cold_start_hit_ratio_at_{}'.format(K), init_dict)
        # this indicates which model to be saved as BEST model
        best_performance_to_save = 'recall_at_20'

        for epoch in range(int(self.config.train.epochs)):
            current_epoch = epoch + self.loaded_epoch + 1
            if current_epoch > int(self.config.train.epochs):
                logger.print('Training completed.')
                break

            if 'NRMS_BERT_EMB_ITER_TRAIN' in self.config.model_config.module_type:
                iter_transformer = 10
                iter_graph = 20
                d = (current_epoch-1) % (iter_transformer+iter_graph)
                self.transformer_optimizer.zero_grad()
                self.graph_optimizer.zero_grad()
                if d < iter_graph:
                    print('using optimizer: graph')
                    self.optimizer = self.graph_optimizer
                else:
                    print('using optimizer: transformer')
                    self.optimizer = self.transformer_optimizer

            #############################################
            #
            #                Train
            #
            #############################################
            # zero the parameter gradients
            self.model.train()
            total_loss_list = []
            batch_cor_list = []
            train_s_t = time.time()
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                if i_batch == 0:
                    self.optimizer.zero_grad()

                train_batch = {
                    'users': sample_batched['users'].to(self.config.device),
                    'pos_items': sample_batched['pos_items'].to(self.config.device),
                    'neg_items': sample_batched['neg_items'].to(self.config.device),
                }

                batch_loss, _, _, batch_cor = self.model(train_batch)

                # Add up batch loss from all GPUs
                total_loss = torch.sum(batch_loss)
                total_loss.backward()
                
                # Keep the avg batch cor
                batch_cor = torch.mean(batch_cor)

                logger.print("epoch {} - batch {} - current loss {}, batch cor {} - {}/{}".format(current_epoch,
                                                                                        i_batch,
                                                                                        total_loss.detach().cpu().numpy(),
                                                                                        batch_cor.detach().cpu().numpy(),
                                                                                        i_batch,
                                                                                        len(self.train_data_loader)))
                
                if i_batch % gradient_accumulation_steps == 0 and i_batch != 0:
                    if self.config.train.additional.gradient_clipping != 0:
                        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                                self.config.train.additional.gradient_clipping)
                    # print('optimizer step!')
                    self.optimizer.step()
                
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        if self.config.train.scheduler not in ['cosine']:
                            self.scheduler.step()
                        # print('scheduler step! LR:', self.scheduler.get_last_lr())
                        # print([group['lr'] for group in self.optimizer.param_groups])

                total_loss_list.append(total_loss.detach().cpu().numpy())
                batch_cor_list.append(batch_cor.detach().cpu().numpy())
                
            train_e_t = time.time()

            # Add to tensorboard
            writer.add_scalar('train/total_loss', np.mean(np.array(total_loss_list)), current_epoch)
            writer.add_scalar('train/batch_cor', np.mean(np.array(batch_cor_list)), current_epoch)
            writer.add_scalar('train/time', (train_e_t - train_s_t), current_epoch)
            
            if self.scheduler:
                if self.config.train.scheduler in ['cosine']:
                    print('scheduler step!', self.scheduler.get_last_lr())
                    self.scheduler.step()

            if self.scheduler:
                for index, group in enumerate(self.optimizer.param_groups):
                    writer.add_scalar('train/param_group_{}_LR'.format(index), 
                            group['lr'], current_epoch)
            else:
                writer.add_scalar('train/param_group_0_LR', 
                            self.config.train.lr, current_epoch)
            writer.flush()
            
            logger.print("epoch {} completed - current loss {}, batch cor {}".format(current_epoch,
                                                                                        np.mean(np.array(total_loss_list)),
                                                                                        np.mean(np.array(batch_cor_list))))
                
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            if current_epoch % save_interval == 0:
                self.save_checkpoint(current_epoch, record_best_model=False)

            #############################################
            #
            #            Validation and Test
            #
            #############################################
            if current_epoch % self.config.valid.step_size == 0:
                if self.config.valid.enable_validation:
                    # Run validation
                    logger.print('epoch {} running validation...'.format(current_epoch))
                    test_s_t = time.time()
                    with torch.no_grad():
                        self.model.eval()
                        ret = self.test(mode='valid')
                    test_e_t = time.time()
                    logger.print('epoch {} validation finished.'.format(current_epoch))
                    train_res = PrettyTable()
                    train_res.field_names = ["Epoch", "training time", "validation time", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
                    train_res.add_row(
                        [epoch, train_e_t - train_s_t, test_e_t - test_s_t, np.mean(np.array(total_loss_list)), ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
                    )
                    print(train_res)
                    
                    # Save best performance and corresponding model
                    for index, K in enumerate(self.config.model_config.Ks):
                        for metrics in ['recall', 'ndcg', 'precision', 'hit_ratio']:
                            metrics_name = '{}_at_{}'.format(metrics, K)
                            metrics_number = ret[metrics][index]
                            # save to best performance if it is better
                            if metrics_number > best_performance[metrics_name]['valid_best']:
                                best_performance[metrics_name].update({'valid_best': metrics_number, 
                                                                    'valid_epoch': current_epoch})
                                # if metrics_name == best_performance_to_save:
                                #     # save this model as the best model
                                #     self.save_checkpoint(current_epoch, record_best_model=True)

                    ### Add test results to tensorboard
                    writer.add_scalar('valid/time', (test_e_t - test_s_t), current_epoch)
                    for index, K in enumerate(self.config.model_config.Ks):
                        writer.add_scalar('valid/recall_at_{}'.format(K), ret['recall'][index], current_epoch)
                        writer.add_scalar('valid/ndcg_at_{}'.format(K), ret['ndcg'][index], current_epoch)
                        writer.add_scalar('valid/precision_at_{}'.format(K), ret['precision'][index], current_epoch)
                        writer.add_scalar('valid/hit_ratio_at_{}'.format(K), ret['hit_ratio'][index], current_epoch)
                
                ####### Test ########
                logger.print('epoch {} running test...'.format(current_epoch))
                test_s_t = time.time()
                with torch.no_grad():
                    self.model.eval()
                    ret = self.test()
                test_e_t = time.time()
                logger.print('epoch {} test finished.'.format(current_epoch))
                train_res = PrettyTable()
                train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
                train_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, np.mean(np.array(total_loss_list)), ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
                )
                print(train_res)
                
                # Save best performance and corresponding model
                for index, K in enumerate(self.config.model_config.Ks):
                    for metrics in ['recall', 'ndcg', 'precision', 'hit_ratio']:
                        metrics_name = '{}_at_{}'.format(metrics, K)
                        metrics_number = ret[metrics][index]
                        # save to best performance if it is better
                        if metrics_number > best_performance[metrics_name]['best']:
                            best_performance[metrics_name].update({'best': metrics_number, 
                                                                'epoch': current_epoch})
                            if metrics_name == best_performance_to_save:
                                # save this model as the best model
                                self.save_checkpoint(current_epoch, record_best_model=True)


                ### Add test results to tensorboard
                writer.add_scalar('test/time', (test_e_t - test_s_t), current_epoch)
                for index, K in enumerate(self.config.model_config.Ks):
                    writer.add_scalar('test/recall_at_{}'.format(K), ret['recall'][index], current_epoch)
                    writer.add_scalar('test/ndcg_at_{}'.format(K), ret['ndcg'][index], current_epoch)
                    writer.add_scalar('test/precision_at_{}'.format(K), ret['precision'][index], current_epoch)
                    writer.add_scalar('test/hit_ratio_at_{}'.format(K), ret['hit_ratio'][index], current_epoch)
                
                #############################################
                #
                #                cold-start test
                #
                #############################################
                if self.config.data_loader.additional.split_cold_start:
                    logger.print('epoch {} running cold-start test...'.format(current_epoch))
                    test_s_t = time.time()
                    with torch.no_grad():
                        self.model.eval()
                        ret = self.test(mode='cold_start_test')
                    test_e_t = time.time()
                    logger.print('epoch {} cold-start test finished.'.format(current_epoch))
                    train_res = PrettyTable()
                    train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
                    train_res.add_row(
                        [epoch, train_e_t - train_s_t, test_e_t - test_s_t, np.mean(np.array(total_loss_list)), ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
                    )
                    print(train_res)
                    
                    # Save best performance and corresponding model
                    for index, K in enumerate(self.config.model_config.Ks):
                        for metrics in ['recall', 'ndcg', 'precision', 'hit_ratio']:
                            metrics_name = 'cold_start_{}_at_{}'.format(metrics, K)
                            metrics_number = ret[metrics][index]
                            # save to best performance if it is better
                            if metrics_number > best_performance[metrics_name]['best']:
                                best_performance[metrics_name] = {'best': metrics_number, 
                                                                    'epoch': current_epoch}
                                

                    ### Add test results to tensorboard
                    writer.add_scalar('cold_start_test/time', (test_e_t - test_s_t), current_epoch)
                    for index, K in enumerate(self.config.model_config.Ks):
                        writer.add_scalar('cold_start_test/recall_at_{}'.format(K), ret['recall'][index], current_epoch)
                        writer.add_scalar('cold_start_test/ndcg_at_{}'.format(K), ret['ndcg'][index], current_epoch)
                        writer.add_scalar('cold_start_test/precision_at_{}'.format(K), ret['precision'][index], current_epoch)
                        writer.add_scalar('cold_start_test/hit_ratio_at_{}'.format(K), ret['hit_ratio'][index], current_epoch)
                    
                    writer.flush()

            #############################################
            #
            #                End of Epoch
            #
            #############################################
            print('=========== best performance start ===========')
            pprint(best_performance)
            logger.print(best_performance)
            self.save_results(best_performance, 'metrics.json')
            print('=========== best performance end ===========')
            logger.print('epoch {} finished.'.format(current_epoch))


    def generate_NRMS_embeddings(self):
        """This function generates NRMS-BERT embeddings from pretrained weights

        Returns:
            Tensor(cpu): all_item_embed
            Tensor(cpu): all_user_embed
        """
        Ks = self.config.model_config.Ks
        result = {'precision': np.zeros(len(Ks)),
                'recall': np.zeros(len(Ks)),
                'ndcg': np.zeros(len(Ks)),
                'hit_ratio': np.zeros(len(Ks)),
                'auc': 0.}

        n_params = self.data_loader.n_params
        train_user_set = self.data_loader.train_user_set
        test_user_set = self.data_loader.test_user_set
        batch_test_flag = self.config.valid.additional.batch_test_flag

        n_items = n_params['n_items']
        n_users = n_params['n_users']
        print(n_params)
        u_batch_size = 64

        # batch size in running GPU forward pass
        i_batch_size = 64 * 8

        # Get all users
        test_users = list(train_user_set.keys())
        if self.config.data_loader.additional.split_cold_start:
            # add cold start users
            test_users += list(self.data_loader.cold_start_train_user_set.keys())
            train_user_set = train_user_set.copy()
            train_user_set.update(self.data_loader.cold_start_train_user_set)
            test_user_set = test_user_set.copy()
            test_user_set.update(self.data_loader.cold_start_test_user_set)
            print('extended to', len(test_users), 'users.')

        # test_users = list(set(list(test_user_set.keys()) + list(train_user_set.keys())))
        n_test_users = len(test_users)
        n_user_batchs = n_test_users // u_batch_size + 1

        count = 0

        # Generate user embeddings
        user_list_batch = test_users[:]
        user_batch = torch.LongTensor(np.array(user_list_batch)).to(self.config.device)

        all_user_embed = []
        for u_start in tqdm(range(0, n_test_users, u_batch_size)):
            u_end = min(u_start + u_batch_size, n_users)
            user_ids = test_users[u_start:u_end]
            user_histories = []
            if not self.config.model_config.use_att_mask:
                for user_id in user_ids:
                    current_user_history = train_user_set[int(user_id)]
                    user_histories.append(random.choices(current_user_history, k=self.config.model_config.num_history))
                # print(user_histories)
                user_histories = torch.LongTensor(user_histories)
                # print(user_histories.shape)
                user_ids = torch.LongTensor(user_ids)
                test_batch = {
                    'users': user_ids.to(self.config.device),
                    'user_histories': user_histories.to(self.config.device),
                    'test_mode': 'user_embed',
                    'compute_loss': False,
                }
            else:
                user_history_att_masks = []
                k=self.config.model_config.num_history
                for user_id in user_ids:
                    exist_history = train_user_set[int(user_id)]
                    history_att_mask = torch.zeros(k)
                    if len(exist_history) <= k:
                        # less than needed
                        user_history = exist_history + [0]*(k-len(exist_history))
                        history_att_mask[:len(exist_history)] = 1
                    elif len(exist_history) > k:
                        # more than needed
                        user_history = random.sample(exist_history, k)
                        history_att_mask[:] = 1
                    user_histories.append(user_history)
                    history_att_mask = history_att_mask.numpy().tolist()
                    user_history_att_masks.append(history_att_mask)
                
                # Transform to tensors
                user_histories = torch.LongTensor(user_histories)
                user_history_att_masks = torch.LongTensor(user_history_att_masks)
                
                user_ids = torch.LongTensor(user_ids)
                test_batch = {
                    'users': user_ids.to(self.config.device),
                    'user_histories': user_histories.to(self.config.device),
                    'user_history_att_masks': user_history_att_masks.to(self.config.device),
                    'test_mode': 'user_embed',
                    'compute_loss': False,
                }
            user_embed = self.NRMS(test_batch)
            all_user_embed.append(user_embed.detach())
        
        # # n_test_users x embed_size
        u_g_embeddings = torch.cat(all_user_embed, axis=0)
        all_user_embed = u_g_embeddings
        print(all_user_embed)
        # u_g_embeddings = torch.ones(n_test_users, 64).to(self.config.device)

        # batch-item test
        n_item_batchs = n_items // i_batch_size + 1
        rate_batch = np.zeros(shape=(len(user_batch), n_items))

        i_count = 0
        all_item_embed = []
        for i_batch_id in tqdm(range(n_item_batchs)):
            i_start = i_batch_id * i_batch_size
            i_end = min((i_batch_id + 1) * i_batch_size, n_items)

            item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end-i_start)
            test_batch = {
                'pos_items': item_batch.to(self.config.device),
                'test_mode': 'item_embed',
                'compute_loss': False,
            }
            item_embed = self.NRMS(test_batch)
            # print(item_embed)
            # i_batch_size x embed_size
            i_g_embddings = item_embed.detach()
            all_item_embed.append(i_g_embddings)
            # Run model rating to get user preference matrix
            i_rate_batch = torch.matmul(u_g_embeddings, i_g_embddings.t()).cpu()
            # print(i_rate_batch)
            # print(i_rate_batch.shape)
            rate_batch[:, i_start: i_end] = i_rate_batch
            i_count += i_rate_batch.shape[1]

        all_item_embed = torch.cat(all_item_embed, axis=0)
        assert i_count == n_items

        
        # Zip data together
        user_batch_rating_uid = zip(rate_batch, user_list_batch)

        # Launch multi-processing to speed up ranking
        manager = Manager()
        recorder_queue = manager.Queue()
        task_queue = manager.Queue(100)
        NUM_PROCESSES = 8 #self.config.test.additional.multiprocessing
        ps = []
        for i in range(NUM_PROCESSES):
            p = spawn(test_thread, args=(i,
                                            self.config,
                                            task_queue,
                                            recorder_queue,
                                            train_user_set,
                                            test_user_set,
                                            n_params,
                                            Ks), join=False)
            ps.append(p)
        
        print('waiting for subprocesses to finish...')
        for i_batch, sample_batched in enumerate(tqdm(user_batch_rating_uid, total=len(user_list_batch))):
            try:
                task_queue.put((i_batch, sample_batched), block=True)
                # print('new task {} has been initialized'.format(i))
                i = i + 1
            except Exception as e:
                print(e)

        # Wait for all processes done
        for p in ps:
            p.join()

        # Read recorder queue until finish all
        batch_result = []
        count_task = 0
        while recorder_queue.qsize() > 0:
            output = recorder_queue.get(block=True)
            # print('getting', len(batch_result), '/', len(user_list_batch))
            batch_result.append(output)
        
        logger.print('Testing finished and data has been collected from the multi-processing unit.')

        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users

        assert count == n_test_users
        print(result)

        return all_item_embed.cpu(), all_user_embed.cpu()




    def test(self, mode='test', save=False):
        """This function tests the model during running

        Args:
            mode (str, optional): 'test'/'cold_start_test'. Defaults to 'test'.
            save (bool, optional): True to save model output embeddings. Defaults to False.
        Returns:
            Dict: dict containing test metrics. Format is similar to 
                result = {'precision': np.zeros(len(Ks)),
                'recall': np.zeros(len(Ks)),
                'ndcg': np.zeros(len(Ks)),
                'hit_ratio': np.zeros(len(Ks)),
                'auc': 0.}
        """


        Ks = self.config.model_config.Ks
        result = {'precision': np.zeros(len(Ks)),
                'recall': np.zeros(len(Ks)),
                'ndcg': np.zeros(len(Ks)),
                'hit_ratio': np.zeros(len(Ks)),
                'auc': 0.}
        n_params = self.data_loader.n_params
        if mode == 'test':
            train_user_set = self.data_loader.train_user_set
            test_user_set = self.data_loader.test_user_set
        elif mode == 'valid':
            train_user_set = self.data_loader.train_user_set
            test_user_set = self.data_loader.valid_user_set
        elif mode == 'cold_start_test':
            train_user_set = self.data_loader.cold_start_train_user_set
            test_user_set = self.data_loader.cold_start_test_user_set

        batch_test_flag = self.config.valid.additional.batch_test_flag

        n_items = n_params['n_items']
        n_users = n_params['n_users']

        # Compute CPU cores
        cores = multiprocessing.cpu_count() // 2

        u_batch_size = self.config.valid.batch_size # Not used

        # batch size in running GPU forward pass
        i_batch_size = self.config.valid.batch_size // 100

        # Get all test users
        test_users = list(test_user_set.keys())
        n_test_users = len(test_users)
        n_user_batchs = n_test_users // u_batch_size + 1

        count = 0

        if self.config.using_data_parallel:
            # Using Mutiple GPUs
            model = self.model.module
        else:
            model = self.model

        # Generate entity and user embeddings
        entity_gcn_emb, user_gcn_emb = model.generate()

        # for u_batch_id in range(n_user_batchs):
            # print(u_batch_id, n_user_batchs)
            # start = u_batch_id * u_batch_size
            # end = (u_batch_id + 1) * u_batch_size

        user_list_batch = test_users[:]
        user_batch = torch.LongTensor(np.array(user_list_batch)).to(self.config.device)
        u_g_embeddings = user_gcn_emb[user_batch]

        if batch_test_flag:
            # batch-item test
            n_item_batchs = n_items // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), n_items))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end-i_start).to(self.config.device)
                i_g_embddings = entity_gcn_emb[item_batch]

                # Run model rating to get user preference matrix
                i_rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == n_items
        else:
            # all-item test
            item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(self.config.device)
            i_g_embddings = entity_gcn_emb[item_batch]
            rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

        # Zip data together
        user_batch_rating_uid = zip(rate_batch, user_list_batch)

        # Launch multi-processing to speed up ranking
        manager = Manager()
        recorder_queue = manager.Queue()
        task_queue = manager.Queue(100)
        NUM_PROCESSES = 8 #self.config.test.additional.multiprocessing
        ps = []
        for i in range(NUM_PROCESSES):
            p = spawn(test_thread, args=(i,
                                            self.config,
                                            task_queue,
                                            recorder_queue,
                                            train_user_set,
                                            test_user_set,
                                            n_params,
                                            Ks), join=False)
            ps.append(p)
        
        print('waiting for subprocesses to finish...')
        for i_batch, sample_batched in enumerate(tqdm(user_batch_rating_uid, total=len(user_list_batch))):
            try:
                task_queue.put((i_batch, sample_batched), block=True)
                # print('new task {} has been initialized'.format(i))
                i = i + 1
            except Exception as e:
                print(e)

        # Wait for all processes done
        for p in ps:
            p.join()

        # Read recorder queue until finish all
        batch_result = []
        count_task = 0
        while recorder_queue.qsize() > 0:
            output = recorder_queue.get(block=True)
            # print('getting', len(batch_result), '/', len(user_list_batch))
            batch_result.append(output)
            # input()
        
        logger.print('Testing finished and data has been collected from the multi-processing unit.')
        
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users

        assert count == n_test_users

        if save:
            """Save ratings of model output to files for evaluation."""
            # == WARNING: File is very large if saving embeddings as well. ==
            to_save_pickle_path = os.path.join(self.config.results_path, 'model_output_{}.pkl'.format(self.loaded_epoch))
            with open(to_save_pickle_path, 'wb') as f:
                data = {
                    'result': result,
                    'rate_batch': rate_batch,
                    'user_list_batch': user_list_batch,
                    # 'entity_gcn_emb':entity_gcn_emb,
                    # 'user_gcn_emb':user_gcn_emb,
                }
                pickle.dump(data, f, protocol=4)
                print('Model output has been saved to {}'.format(to_save_pickle_path))

        return result



def test_thread(thread_subid, thread_index, config, 
                                            task_queue,
                                             recorder_queue, 
                                             train_user_set, 
                                             test_user_set, 
                                             n_params, 
                                             Ks):
    """Testing thread for conducting multi-processing

    Args:
        thread_subid ([type]): [description]
        thread_index ([type]): [description]
        config ([type]): [description]
        task_queue ([type]): [description]
        recorder_queue ([type]): [description]
        train_user_set ([type]): [description]
        test_user_set ([type]): [description]
        n_params ([type]): [description]
        Ks ([type]): [description]
    """
    try:
        # print('start!')
        # Set seed
        # if config.seed:
        #     # set_seed(config.seed)
        #     print("thread SEED is set to:", config.seed)
        print('thread {} initiated'.format(thread_index))
        time.sleep(20)
        RETRY = 0
        while True:
            if task_queue.qsize() == 0:
                if RETRY < 3:
                    print('thread {} retrying... {}'.format(thread_index, RETRY))
                    time.sleep(3)
                    RETRY += 1
                else:
                    break
            else:
                RETRY = 0
            
            CONTINUE = True
            try:
                i_batch, sample_batched = task_queue.get(block=False)
                # print('thread {} gets task {}'.format(thread_index, i_batch))
            except Exception as e:
                print(e)
                CONTINUE = False

            if CONTINUE:
                x = sample_batched
                # user u's ratings for user u
                rating = x[0]
                # uid
                u = x[1]
                # user u's items in the training set
                n_items = n_params['n_items']
                try:
                    training_items = train_user_set[u]
                except Exception:
                    training_items = []
                # user u's items in the test set
                user_pos_test = test_user_set[u]

                all_items = set(range(0, n_items))

                test_items = list(all_items - set(training_items))

                if config.model_config.test_flag == 'part':
                    r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
                else:
                    r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)
                
                log_result = get_performance(user_pos_test, r, auc, Ks)
                # print(log_result)
                recorder_queue.put(log_result)

    except Exception as e:
        print(e)
    
    print('thread {} finished'.format(thread_index))