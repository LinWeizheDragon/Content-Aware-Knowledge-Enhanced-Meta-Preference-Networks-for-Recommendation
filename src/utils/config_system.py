"""
config_system.py:  
    Functions for initializing config
        - Read json/jsonnet config files
        - Parse args and override parameters in config files
"""

__author__ = "Weizhe Lin"
__copyright__ = "Copyright 2021, Weizhe Lin"
__version__ = "1.0.0"
__email__ = "wl356@cam.ac.uk"
__status__ = "Published for Github"

import os
import shutil
import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler

import json
import _jsonnet
import datetime
import time
from easydict import EasyDict
from pprint import pprint
import time
from utils.dirs import create_dirs
from pathlib import Path

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided

    try:
        config_dict = json.loads(_jsonnet.evaluate_file(json_file))
        # EasyDict allows to access dict values as attributes (works recursively).
        config = EasyDict(config_dict)
        return config, config_dict
    except ValueError:
        print("INVALID JSON file.. Please provide a good json file")
        exit(-1)

def process_config(args):
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    path = Path(script_dir).parent
    config, _ = get_config_from_json(args.config)

    # Some default paths
    if not config.DATA_FOLDER:
        # Default path
        config.DATA_FOLDER = os.path.join(str(path), 'Data')
    if not config.EXPERIMENT_FOLDER:
        # Default path
        config.EXPERIMENT_FOLDER = os.path.join(str(path), 'Experiments')
    if not config.TENSORBOARD_FOLDER:
        # Default path
        config.TENSORBOARD_FOLDER = os.path.join(str(path), 'Data_TB', 'tb_logs')

    
    
    # Override using passed parameters
    config.cuda = not args.disable_cuda
    config.gpu_device = args.device
    # if args.device != -1:
    #     config.gpu_device = args.device
    config.reset = args.reset
    config.mode = args.mode
    if args.experiment_name != '':
        config.experiment_name = args.experiment_name
    config.model_config.module_type = args.module_type
    config.data_loader.additional.regenerate_data = args.regenerate_data
    config.data_loader.additional.regenerate_transformer_data = args.regenerate_transformer_data
    config.data_loader.dummy_dataloader = args.dummy_dataloader
    config.data_loader.negative_sampling_mode = args.negative_sampling_mode
    if args.dataset_name != 'default':
        config.data_loader.dataset_name = args.dataset_name
    if args.load_num_movies != -1:
        config.data_loader.additional.load_num_movies = args.load_num_movies
    if args.load_num_users != -1:
        config.data_loader.additional.load_num_users = args.load_num_users
    if args.epochs != -1:
        config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.scheduler = args.scheduler
    config.train.lr = args.lr
    config.train.additional.gradient_clipping = args.clipping
    if args.test_num_evaluation != -1:
        config.test.num_evaluation = args.test_num_evaluation
    if args.test_batch_size != -1:
        config.test.batch_size = args.test_batch_size
    if args.test_evaluation_name:
        config.test.evaluation_name = args.test_evaluation_name
    if args.test_num_processes != -1:
        config.test.additional.multiprocessing = args.test_num_processes

    if config.mode == "train":
        config.train.load_best_model = args.load_best_model
        config.train.load_model_path = args.load_model_path
        config.train.load_epoch = args.load_epoch
    else:
        config.train.load_best_model = args.load_best_model
        config.train.load_model_path = args.load_model_path
        config.train.load_epoch = args.load_epoch
        config.test.load_best_model = args.load_best_model
        config.test.load_model_path = args.load_model_path
        config.test.load_epoch = args.load_epoch

    
    # Loading model parameters from args
    # ===== KGIN ===== #
    config.model_config.dim = args.dim
    config.model_config.l2 = args.l2
    config.model_config.sim_regularity = args.sim_regularity
    config.model_config.inverse_r = args.inverse_r
    config.model_config.node_dropout = args.node_dropout
    config.model_config.node_dropout_rate = args.node_dropout_rate
    config.model_config.mess_dropout = args.mess_dropout
    config.model_config.mess_dropout_rate = args.mess_dropout_rate
    config.model_config.channel = args.channel
    config.model_config.Ks = args.Ks
    config.model_config.test_flag = args.test_flag
    config.valid.additional.batch_test_flag = args.batch_test_flag
    config.test.additional.batch_test_flag = args.batch_test_flag
    config.model_config.n_factors = args.n_factors
    config.model_config.ind = args.ind
    config.model_config.context_hops = args.context_hops

    ###### My model
    config.model_config.wl_test_iter = args.wl_test_iter
    config.model_config.freeze_transformer_layers = args.freeze_transformer_layers
    config.model_config.num_negative_samples = args.num_negative_samples
    config.model_config.num_history = args.num_history
    config.model_config.load_transformer_path = args.load_transformer_path
    config.model_config.load_graph_path = args.load_graph_path
    config.model_config.freeze_graph = args.freeze_graph
    config.model_config.split_model_on_gpus = args.split_model_on_gpus
    config.model_config.pca_ratio = args.pca_ratio
    config.model_config.cross_system_loss_decay = args.cross_system_loss_decay
    config.data_loader.additional.split_cold_start = args.split_cold_start
    config.data_loader.additional.cold_start_ratio = args.cold_start_ratio
    config.model_config.num_meta_preferences = args.num_meta_preferences
    config.model_config.use_att_mask = args.use_att_mask
    config.data_loader.additional.extend_cbf_string = args.extend_cbf_string
    config.valid.enable_validation = args.enable_validation

    # Generated Paths
    config.log_path = os.path.join(config.EXPERIMENT_FOLDER, config.experiment_name, config.mode)
    config.experiment_path = os.path.join(config.EXPERIMENT_FOLDER, config.experiment_name)
    config.saved_model_path = os.path.join(config.EXPERIMENT_FOLDER, config.experiment_name, "train", 'saved_model')
    if config.mode == "train":
        config.imgs_path = os.path.join(config.EXPERIMENT_FOLDER, config.experiment_name, "train", 'imgs')
    else:
        config.imgs_path = os.path.join(config.EXPERIMENT_FOLDER, config.experiment_name, "test",
                                        config.test.evaluation_name, 'imgs')
        config.results_path = os.path.join(config.EXPERIMENT_FOLDER, config.experiment_name, "test",
                                        config.test.evaluation_name)
    config.tensorboard_path = os.path.join(config.TENSORBOARD_FOLDER, config.experiment_name)

    if args.job_id:
        # Under ITP, output TB data to specific folder!
        config.ITP_TENSORBOARD_FOLDER = os.path.join("/home/v-weizhelin/tensorboard", args.job_id, 'tb_logs')
        config.ITP_tensorboard_path = os.path.join(config.ITP_TENSORBOARD_FOLDER, config.experiment_name)
    else:
        config.ITP_TENSORBOARD_FOLDER = ""
        config.ITP_tensorboard_path = ""

    return config



