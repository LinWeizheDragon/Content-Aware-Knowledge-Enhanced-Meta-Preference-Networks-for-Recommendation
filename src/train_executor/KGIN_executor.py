"""
KGIN_executor.py:  Training functions for KGIN
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


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

from models.KGIN.KGIN import Recommender
from utils.KGIN_evaluate import ranklist_by_heapq, get_auc, ranklist_by_sorted, get_performance
from transformers import get_linear_schedule_with_warmup


class KGINExecutor(BaseExecutor):
    def __init__(self, config, data_loader):
        BaseExecutor.__init__(self, config, data_loader)
        logger.print('Training has been initiated.')

        """define model"""
        self.model = Recommender(
            config,
            data_loader.n_params,
            data_loader.graph,
            data_loader.mean_mat_list[0]).to(config.device)

        
        if self.config.using_data_parallel:
            # Using Mutiple GPUs
            self.model = nn.DataParallel(self.model)

        self.model.to(self.config.device)
        
        print(self.model)

        """define optimizer"""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.train.lr)

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
        for index, K in enumerate(self.config.model_config.Ks):
            best_performance.setdefault('recall_at_{}'.format(K), {'best': 0, 'epoch': -1})
            best_performance.setdefault('ndcg_at_{}'.format(K), {'best': 0, 'epoch': -1})
            best_performance.setdefault('precision_at_{}'.format(K), {'best': 0, 'epoch': -1})
            best_performance.setdefault('hit_ratio_at_{}'.format(K), {'best': 0, 'epoch': -1})
        # this indicates which model to be saved as BEST model
        best_performance_to_save = 'recall_at_20'

        for epoch in range(int(self.config.train.epochs)):
            current_epoch = epoch + self.loaded_epoch + 1
            if current_epoch > int(self.config.train.epochs):
                logger.print('Training completed.')
                break
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

                # print(batch_loss, batch_cor)

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
            #                Validation
            #
            #############################################
            if current_epoch % self.config.valid.step_size == 0:
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
                            best_performance[metrics_name] = {'best': metrics_number, 
                                                                'epoch': current_epoch}
                            if metrics_name == best_performance_to_save:
                                # save this model as the best model
                                if current_epoch % save_interval == 0:
                                    self.save_checkpoint(current_epoch, record_best_model=True)


                ### Add test results to tensorboard
                writer.add_scalar('test/time', (test_e_t - test_s_t), current_epoch)
                for index, K in enumerate(self.config.model_config.Ks):
                    writer.add_scalar('test/recall_at_{}'.format(K), ret['recall'][index], current_epoch)
                    writer.add_scalar('test/ndcg_at_{}'.format(K), ret['ndcg'][index], current_epoch)
                    writer.add_scalar('test/precision_at_{}'.format(K), ret['precision'][index], current_epoch)
                    writer.add_scalar('test/hit_ratio_at_{}'.format(K), ret['hit_ratio'][index], current_epoch)
                
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

    def test(self):
        """This function tests the model during running

        Args:
            mode (str, optional): 'test'/'cold_start_test'. Defaults to 'test'.
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
        train_user_set = self.data_loader.train_user_set
        test_user_set = self.data_loader.test_user_set
        batch_test_flag = self.config.valid.additional.batch_test_flag

        n_items = n_params['n_items']
        n_users = n_params['n_users']

        # Compute CPU cores
        cores = multiprocessing.cpu_count() // 2

        u_batch_size = self.config.valid.batch_size # Not used

        # batch size in running GPU forward pass
        i_batch_size = self.config.valid.batch_size 

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
        # input('read finished!')


        # test_one_user_func=partial(test_one_user, 
        #                     config=self.config,
        #                     train_user_set=train_user_set,
        #                     test_user_set=test_user_set,
        #                     n_params=n_params,
        #                     Ks=Ks)
        
        # batch_result = []
        # for batch_output in tqdm(pool.imap_unordered(test_one_user_func, user_batch_rating_uid), total=len(user_list_batch)):
        #     batch_result.append(batch_output)
        
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users

        assert count == n_test_users
        # pool.close()
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