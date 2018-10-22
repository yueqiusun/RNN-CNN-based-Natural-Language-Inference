# First lets improve libraries that we are going to be used in this lab session
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
import pickle as pkl
import random
import pdb
random.seed(134)

from bagofwords import BagOfWords
from hyperparameter import Hyperparameter as hp


import re
 
from train_f import *
%load_ext autoreload
%autoreload 2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

PAD_IDX = 0
UNK_IDX = 1
BATCH_SIZE = 100