import os
import numpy as np
import torch
import importlib
import consts
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from scipy.spatial.distance import cdist
from scipy.stats import ortho_group
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pyLDAvis
import random
from toolbox.alias_multinomial import AliasMultinomial
from nltk.tokenize import word_tokenize
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import roc_auc_score, mean_squared_error

# 重新載入 consts 以確保變數最新
importlib.reload(consts)
torch.manual_seed(consts.SEED)
np.random.seed(consts.SEED)

