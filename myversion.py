import logging
import os

import numpy as np
import pyLDAvis
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from scipy.spatial.distance import cdist
from scipy.stats import ortho_group
from sklearn.metrics import roc_auc_score
from torch.nn import Parameter
from sklearn.metrics import roc_auc_score, mean_squared_error

from wordcloud import WordCloud
import matplotlib.pyplot as plt

import optuna

from dataset.csv import CSVDataset
from toolbox.helper_functions import get_dataset