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

torch.manual_seed(consts.SEED)
np.random.seed(consts.SEED)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from dataset.csv import CSVDataset
from toolbox.helper_functions import get_dataset

class GDCM(nn.Module):
    def __init__(self, out_dir, embed_dim=300, nnegs=15, nconcepts=25, lam=100.0, rho=100.0, eta=1.0,
                 doc_concept_probs=None, word_vectors=None, theta=None, gpu=None,
                 inductive=True, inductive_dropout=0.01, hidden_size=100, num_layers=1,
                 bow_train=None, y_train=None, bow_test=None, y_test=None, doc_windows=None, vocab=None,
                 word_counts=None, doc_lens=None, expvars_train=None, expvars_test=None, file_log=False, norm=None):
        