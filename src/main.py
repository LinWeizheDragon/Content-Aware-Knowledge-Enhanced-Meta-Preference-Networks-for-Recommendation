"""
main.py:  
    Start functions
        - Read json/jsonnet config files
        - Parse args and override parameters in config files
        - Find selected data loader and initialize
        - Run TrainExecutor to perform training and testing
"""

__author__ = "Weizhe Lin"
__copyright__ = "Copyright 2021, Weizhe Lin"
__version__ = "1.0.0"
__email__ = "wl356@cam.ac.uk"
__status__ = "Published for Github"


import os
import argparse
import torch

import json
from pprint import pprint
from utils.config_system import process_config
from utils.log_system import logger
from utils.dirs import *
from utils.cuda_stats import print_cuda_statistics
from utils.seed import set_seed


from data_loader_manager import *
from train_executor import *

def main(config):
    pprint(config)

    if config.seed:
        set_seed(config.seed)
        logger.print('All seeds have been set to', config.seed)
        
    if config.mode == 'train':

        DataLoaderWrapper = globals()[config.data_loader.type]
        if DataLoaderWrapper is not None:
            # init data loader
            data_loader = DataLoaderWrapper(config)
        else:
            print('data loader', config.data_loader.type, 'not found!')
            data_loader = None
        
        # init train excecutor
        Train_Executor = globals()[config.train.type]
        executor = Train_Executor(config, data_loader)
        # After Initialization, save config files
        with open(os.path.join(config.experiment_path, "config.jsonnet"), 'w') as config_file:
            save_config = config.copy()
            save_config.pop('device') # Not serialisable
            json.dump(save_config, config_file, indent=4)
            print('config file was successfully saved to', config.experiment_path, 'for future use.')
        # Start training
        executor.train()
    
    else:
        DataLoaderWrapper = globals()[config.data_loader.type]
        if DataLoaderWrapper is not None:
            # init data loader
            data_loader = DataLoaderWrapper(config)
        else:
            print('data loader', config.data_loader.type, 'not found!')
            data_loader = None
        
        # init train excecutor
        Train_Executor = globals()[config.train.type]
        executor = Train_Executor(config, data_loader)
        # Start testing
        executor.run_test()

    #
    # elif config.mode == 'test':
    #     if DataLoaderWrapper is not None:
    #         # init data loader
    #         data_loader = DataLoaderWrapper(config)
    #     else:
    #         data_loader = None
    #     # init test executor
    #     evaluator = Test_Evaluator(config, data_loader)
    #     # Run Evaluation
    #     evaluator.evaluate()

def initialization(args):
    assert args.mode in ['train', 'test', 'run']
    # ===== Process Config =======
    config = process_config(args)
    print(config)
    if config is None:
        return None
    # Create Dirs
    dirs = [
        config.log_path,
    ]
    if config.mode == 'train':
        dirs += [
            config.saved_model_path,
            config.imgs_path,
            config.tensorboard_path
        ]
    if config.mode == 'test':
        dirs += [
            config.imgs_path,
            config.results_path,
        ]

    if config.reset and config.mode == "train":
        # Reset all the folders
        print("You are deleting following dirs: ", dirs, "input y to continue")
        if input() == 'y':
            for dir in dirs:
                try:
                    delete_dir(dir)
                except Exception as e:
                    print(e)
            # Reset load epoch after reset
            config.train.load_epoch = 0
        else:
            print("reset cancelled.")

    create_dirs(dirs)
    print(dirs)
    logger.init_logger(config)

    # set cuda flag
    is_cuda = torch.cuda.is_available()
    if is_cuda and not config.cuda:
        logger.print("WARNING: You have a CUDA device, so you should probably enable CUDA")

    cuda = is_cuda & config.cuda

    if cuda:
        # Using GPUs
        # This is the main GPU device
        config.device = torch.device("cuda:0")
        # torch.cuda.set_device(config.gpu_device)

        # This lists all gpu devices being used
        config.cuda_device = [i for i in range(torch.cuda.device_count())]
        if len(config.cuda_device) > 1:
            # Multiple GPUs
            config.using_data_parallel = True
        else:
            config.using_data_parallel = False
        # print(config.cuda_device)
        # input()
        logger.print("Program will run on *****GPU-CUDA***** ")
        print_cuda_statistics()
    else:
        config.cuda_device = []
        config.device = torch.device("cpu")
        config.using_data_parallel = False
        logger.print("Program will run on *****CPU*****\n")

    logger.print('Initialization done with the config:', str(config))
    return config

def parse_args_sys(args_list=None):
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    arg_parser.add_argument('--DATA_FOLDER', type=str, default='', help='The path to data.')
    arg_parser.add_argument('--EXPERIMENT_FOLDER', type=str, default='', help='The path to save experiments.')
    arg_parser.add_argument('--job_id', type=str, default='')
    arg_parser.add_argument('--disable_cuda', action='store_true', default=False, help='Enable to run on CPU.')
    arg_parser.add_argument('--device', type=int, nargs="+", default=[-1], help='Specify GPU devices to use. -1 for default (all GPUs).')
    arg_parser.add_argument('--module_type', type=str, nargs="+", default=[], help='Select modules for models. See training scripts for examples.')

    arg_parser.add_argument('--mode', type=str, default='', help='train/test')
    arg_parser.add_argument('--reset', action='store_true', default=False, help='Reset the corresponding folder under the experiment_name')
    arg_parser.add_argument('--dummy_dataloader', action='store_true', default=False)
    arg_parser.add_argument('--regenerate_data', action='store_true', default=False, help='Regenerate movie dataset pre-processed data.')
    arg_parser.add_argument('--regenerate_transformer_data', action='store_true', default=False, help='Regenerate pre-trained user/item embeddings from NRMS-BERT for CKMPN training.')
    
    arg_parser.add_argument('--experiment_name', type=str, default='', help='Experiment will be saved under /path/to/EXPERIMENT_FOLDER/$experiment_name$.')
    arg_parser.add_argument('--fp16', action='store_true', default=False, help='Not used.')

    arg_parser.add_argument('--load_best_model', action='store_true', default=False, help='Whether to load model_best.pth.tar.')
    arg_parser.add_argument('--load_epoch', type=int, default=-1, help='Specify which epoch to load.')
    arg_parser.add_argument('--load_model_path', type=str, default="", help='Specify the path of model to load from')

    arg_parser.add_argument("--dataset_name", nargs="?", default="default", help="dataset name")
    arg_parser.add_argument('--load_num_movies', type=int, default=-1, help='Limit of #movies to load (in pre-processing)')
    arg_parser.add_argument('--load_num_users', type=int, default=-1, help='Limit of #users to load (in pre-processing)')

    arg_parser.add_argument('--load_transformer_path', type=str, default='', help='Load NRMS-BERT model for pre-extraction of user/item embeddings.')
    arg_parser.add_argument('--load_graph_path', type=str, default='', help='Not presented in this paper.')
    
    # ===== Testing Configuration ===== #
    arg_parser.add_argument('--test_num_evaluation', type=int, default=-1)
    arg_parser.add_argument('--test_batch_size', type=int, default=-1)
    arg_parser.add_argument('--test_num_processes', type=int, default=-1)
    arg_parser.add_argument('--test_evaluation_name', type=str, default="")

    # ===== Training Configuration ===== #
    arg_parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    arg_parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    arg_parser.add_argument('--epochs', type=int, default=-1, help='number of epochs for training')
    arg_parser.add_argument('--scheduler', type=str, default="none", help='which scheduler to use: [none, linear, cosine]')
    arg_parser.add_argument('--clipping', type=float, default=0, help='gradient clipping')
    arg_parser.add_argument('--negative_sampling_mode', type=str, default='', help='choose from [inversed_ratio, ...]')
    
    # ===== KGIN models ===== #
    arg_parser.add_argument('--dim', type=int, default=64, help='embedding size')
    arg_parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    arg_parser.add_argument('--sim_regularity', type=float, default=1e-4, help='regularization weight for latent factor')
    arg_parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    arg_parser.add_argument("--node_dropout", default=False, action='store_true', help="consider node dropout or not")
    arg_parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    arg_parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    arg_parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    arg_parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    arg_parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    arg_parser.add_argument('--Ks', type=int, nargs='+', default=[20, 40, 60, 80, 100], help='Compute Metrics@K for K in Ks')
    arg_parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    arg_parser.add_argument("--n_factors", type=int, default=4, help="number of latent factor for user favour")
    arg_parser.add_argument("--ind", type=str, default='distance', help="Independence modeling: mi, distance, cosine")
    # ===== KGIN relation context ===== #
    arg_parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')



    # ===== KMPN/CKMPN/NRMS_BERT models ===== #
    arg_parser.add_argument('--wl_test_iter', type=int, default=5,
                        help='#iterations of WL test')
    arg_parser.add_argument('--freeze_transformer_layers', type=int, default=9,
                        help='#layers to be fixed')
    arg_parser.add_argument('--num_negative_samples', type=int, default=4,
                        help='#negative samples used in NRMS-BERT training.')
    arg_parser.add_argument('--num_history', type=int, default=10,
                        help='#user history used in training')
    arg_parser.add_argument('--freeze_graph', action='store_true', default=False)
    arg_parser.add_argument('--split_model_on_gpus', action='store_true', default=False)
    arg_parser.add_argument('--pca_ratio', type=float, default=0.5, help='Ratio of components to keep after PCA reduction before calculating Distance Correlation Loss.')
    arg_parser.add_argument('--cross_system_loss_decay', type=float, default=0.1, help='Loss weight of Cross System Contrastive Loss.')
    arg_parser.add_argument('--num_meta_preferences', type=int, default=64, help='Number of meta preferences for modelling users.')
    
    arg_parser.add_argument('--use_att_mask', action='store_true', default=False, help='NRMS-BERT: False (recommended): randomly sample #history from user histories regardless of duplications; True: for users with less than #history, put attention masks in attention pooling stage.')
    arg_parser.add_argument('--extend_cbf_string', type=bool, default=False)

    # For movies only
    arg_parser.add_argument('--split_cold_start', action='store_true', default=False, help='Whether to split a Cold Start user set from the original dataset.')
    arg_parser.add_argument('--cold_start_ratio', type=float, default=0.03, help='Ratio of users to be put into Cold Start user set.')
    arg_parser.add_argument('--enable_validation', action='store_true', default=False, help='Whether to enable validation (using validation set). Only available in movie datasets.')


    if args_list is None:
        args = arg_parser.parse_args()
    else:
        args = arg_parser.parse_args(args_list)
    return args

if __name__ == '__main__':
    args = parse_args_sys()
    config = initialization(args)
    if config is None:
        exit(0)
    main(config)