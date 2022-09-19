import sys
import time
import json
import copy
from tqdm import tqdm

import json
import torch
from copy import deepcopy
import numpy as np


class DataLoaderWrapper():
    '''
    Data loader wrapper, general class definitions
    '''

    def __init__(self, config):
        self.config = config


    def set_dataloader(self):
        dataset = Dataset(self.config)



class Dataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        return 0