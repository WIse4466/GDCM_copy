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
import consts
from toolbox.alias_multinomial import AliasMultinomial

import optuna

torch.manual_seed(consts.SEED)
np.random.seed(consts.SEED)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

"""
創立一個class GuideedDiverseConceptMiner繼承自pytorch的nn.Module

一、nn.Module的作用
此神經網路的基類提供了一些功能
1. 自動管理模型中的參數(如權重和偏置)
2. 支援模型參數的保存和加載
3. 提供前向傳播(forward)的接口，可以在子類中實現自定義的前向計算

二、繼承的功能
模型參數管理 : 可以使用self.parameters()或self.named_parameters()獲取模型的所有可訓練參數
設備管理 : 可以將模型移動到不同的裝置(如CPU或GPU)，使用.to(device)方法
儲存與加載 : 使用torch.save()和torch.load()輕鬆保存和恢復模型的狀態字典(state_dict)
自定義前向傳播 : 必須在子類中實現forward方法來定義模型的計算過程

三、如何使用繼承的功能
應至少實現
__init()__ : 初始化模型的結構和參數
forward() : 定義輸入數據如何經過模型計算輸出
"""
class GuidedDiverseConceptMiner(nn.Module):

    def __init__(self, out_dir, embed_dim=300, nnegs=15, nconcepts=25, lam=100.0, rho=100.0, eta=1.0,
                 doc_concept_probs=None, word_vectors=None, theta=None, gpu=None,
                 inductive=True, inductive_dropout=0.01, hidden_size=100, num_layers=1,
                 bow_train=None, y_train=None, bow_test=None, y_test=None, doc_windows=None, vocab=None,
                 word_counts=None, doc_lens=None, expvars_train=None, expvars_test=None, file_log=False, norm=None):
        """A class representing a Focused Concept Miner which can mine concepts from unstructured text data while
        making high accuracy predictions with the mined concepts and optional structured data.

        Parameters
        ----------
        out_dir : str
            The directory to save output files from this instance
            檔案輸出的路徑
        embed_dim : int
            The size of each word/concept embedding vector
            每個字/概念的embedding長度(E)
        nnegs : int
            The number of negative context words to be sampled during the training of word embeddings
            在訓練時，要從語料庫裡抽出幾個負樣本
        nconcepts : int
            The number of concepts
            設定概念的數量
        lam : float
            Dirichlet loss weight. The higher, the more sparse is the concept distribution of each document
            控制稀疏程度的參數。越大，概念在每個文件中的分布越分散。公式5
        rho : float
            Prediction loss weight. The higher, the more does the model focus on prediction accuracy
            控制注重預測的參數。越大，模型越注重模型的accuracy。公式8
        eta : float
            Diversity loss weight. The higher, the more different are the concept vectors from each other
            控制多樣性的參數。越大，每個概念向量間彼此就越不同。公式6
        doc_concept_probs [OPTIONAL] : ndarray, shape (n_train_docs, n_concepts)
            Pretrained concept distribution of each training document
            給每個文件預先訓練好的概念分布
        word_vectors [OPTIONAL] : ndarray, shape (vocab_size, embed_dim)
            Pretrained word embedding vectors
            預訓練好的word embedding向量
        theta [OPTIONAL] : ndarray, shape (n_concepts + 1) if `expvars_train` and `expvars_test` are None,
                            or (n_concepts + n_features + 1) `expvars_train` and `expvars_test` are not None
            Pretrained linear prediction weights
            預訓練好的權重。
            例如個概念的權重+1
            例如個概念的權重+各可解釋特徵的權重+1
        gpu [OPTIONAL] : int
            CUDA device if CUDA is available
            有沒有CUDA可以使用GPU
        inductive : bool
            Whether to use neural network to inductively predict the concept weights of each document,
            or use a concept weights embedding
            要使用神經網路來歸納每個文件的概念權重，還是使用concept weights embedding
        inductive_dropout : float
            The dropout rate of the inductive neural network
            歸納概念的神經網路dropout率要設定多少
        hidden_size : int
            The size of the hidden layers in the inductive neural network
            歸納的神經網路其隱藏層有多大
        num_layers : int
            The number of layers in the inductive neural network
            歸納的神經網路有幾層
        bow_train : ndarray, shape (n_train_docs, vocab_size)
            Training corpus encoded as a bag-of-words matrix, where n_train_docs is the number of documents
            in the training set, and vocab_size is the vocabulary size.
            訓練語料庫，使用詞袋模型（Bag-of-Words）編碼為矩陣，其中 n_train_docs 是訓練集的文件數量，vocab_size 是詞彙表的大小。
        y_train : ndarray, shape (n_train_docs,)
            Labels in the training set, ndarray with binary, multiclass, or continuos values.
            訓練集的標籤，可為二元分類、多類別分類或連續值的數組。
        bow_test : ndarray, shape (n_test_docs, vocab_size)
            Test corpus encoded as a matrix
            測試語料庫，使用詞袋模型編碼為矩陣。
        y_test : ndarray, shape (n_test_docs,)
            Labels in the test set, ndarray with binary, multiclass, or continuos values.
            測試集的標籤，可為二元分類、多類別分類或連續值的數組。
        doc_windows : ndarray, shape (n_windows, windows_size + 3)
            Context windows constructed from `bow_train`. Each row represents a context window, consisting of
            the document index of the context window, the encoded target words, the encoded context words,
            and the document's label.
            從 bow_train 構建的上下文窗口（Context Windows）。每一行代表一個上下文窗口，包括文件索引、目標詞的編碼、上下文詞的編碼，以及該文件的標籤。
        vocab : array-like, shape `vocab_size`
            List of all the unique words in the training corpus. The order of this list corresponds
            to the columns of the `bow_train` and `bow_test`
            訓練語料庫中的所有唯一詞彙列表。該列表的順序與 bow_train 和 bow_test 的列順序相對應。
        word_counts : ndarray, shape (vocab_size,)
            The count of each word in the training documents. The ordering of these counts
            should correspond with `vocab`.
            訓練文件中每個詞的出現次數，其排序應與 vocab 對應。
        doc_lens : ndarray, shape (n_train_docs,)
            The length of each training document.
            訓練集中每篇文件的長度（詞數）。
        expvars_train [OPTIONAL] : ndarray, shape (n_train_docs, n_features)
            Extra features for making prediction during the training phase
            訓練階段中用於預測的額外特徵數據集。
        expvars_test [OPTIONAL] : ndarray, shape (n_test_docs, n_features)
            Extra features for making prediction during the testing phase
            測試階段中用於預測的額外特徵數據集。
        file_log : bool
            Whether writes logs into a file or stdout
            是否將日誌寫入檔案（啟用時）或輸出到標準輸出（關閉時）。
        norm : string
            Normalization method to apply on the label data (Y variable) if they are continuous (default: 'standard')
            當標籤資料（Y 變數）為連續型時，應用的正規化方法（預設為 'standard'）。
        """
        super(GuidedDiverseConceptMiner, self).__init__() #呼叫父類別的初始化方法，確保任何從父類繼承的屬性或功能也能正確初始化。
        ndocs = bow_train.shape[0] #ndocs: 訓練文件的數量，從 bow_train 的第一維度獲得。
        vocab_size = bow_train.shape[1] #vocab_size: 詞彙表的大小，從 bow_train 的第二維度獲得。
        self.out_dir = out_dir #設定並創建輸出目錄 out_dir，如果目錄已存在則跳過創建。
        os.makedirs(out_dir, exist_ok=True)
        self.concept_dir = os.path.join(out_dir, "concept") #self.concept_dir: 用於存儲「概念」相關數據的子目錄。
        os.makedirs(self.concept_dir, exist_ok=True)
        self.model_dir = os.path.join(out_dir, "model") #self.model_dir: 用於存儲模型相關數據的子目錄。
        os.makedirs(self.model_dir, exist_ok=True)
        self.embed_dim = embed_dim #嵌入向量的維度大小。
        self.nnegs = nnegs #負樣本數量，用於對比學習或負例採樣。
        self.nconcepts = nconcepts #概念的數量（可能是模型生成的主題數）。
        self.lam = lam #模型的超參數，用於控制正則化或其他學習目標的權重。
        self.rho = rho
        self.eta = eta
        self.alpha = 1.0 / nconcepts #概念的平滑參數（計算為概念數的倒數）。
        self.expvars_train = expvars_train #訓練與測試階段的額外特徵數據集，這些特徵可能用於增強預測能力。
        self.expvars_test = expvars_test
        # print(self.expvars_train[0])
        self.inductive = inductive #表示是否啟用歸納學習模式。這可能影響模型的學習方式。
        if torch.cuda.is_available(): #如果 CUDA (GPU) 可用，使用 cuda。
                self.device = "cuda" 
        elif torch.backends.mps.is_available(): #如果 Apple 的 Metal Performance Shaders (MPS) 可用，使用 mps。
           self.device = "mps"
        else: #否則，默認使用 CPU (cpu)。
            self.device = 'cpu'
           
        
        device = torch.device(self.device) #根據 self.device 的值，指定張量應該在哪個設備（CPU、CUDA 或其他支持的加速器）上運行。
        self.bow_train = torch.tensor(bow_train, dtype=torch.float32, requires_grad=False, device=device)
        """
        bow_train：
            是訓練資料的 Bag-of-Words (BoW) 表示，通常為一個稀疏矩陣，每行表示一個文件，每列表示詞彙的出現次數。
            轉換為 PyTorch 張量：
            dtype=torch.float32：將數據類型設為浮點數。
            requires_grad=False：表示該張量不需要參與梯度計算（不會進行參數更新）。
            device=device：將張量移到指定設備上（如 GPU 或 CPU）。
            self.bow_train：
            儲存轉換後的訓練 BoW 張量。
        """
        assert not (self.inductive and self.bow_train is None)
        """
        self.inductive：
            表示是否處於歸納模式（通常是模型需要從有限的數據中學習概念並推廣到新數據）。
        檢查條件：
            如果模型是歸納模式（self.inductive == True），則必須確保 self.bow_train 不為空。
            如果條件不成立，則觸發 AssertionError，提醒用戶提供完整的訓練數據。
        """
        self.y_train = y_train #將原始的 y_train 數據直接儲存到 self.y_train，後續可能會進行進一步處理或計算。y_train是訓練數據的標籤（目標值），可以是二進制、多類別分類標籤，或連續值（回歸）。
        self.bow_test = torch.tensor(bow_test, dtype=torch.float32, requires_grad=False, device=device) #與 bow_train 的處理方式相同，轉換為 PyTorch 張量，且移到指定設備上。
        self.y_test = y_test #self.y_test儲存測試數據的標籤，類似於 self.y_train。

        """code optimization task 1 (normalization)""" #目標：優化代碼以處理標籤的正規化需求，特別是針對連續型標籤（非二元或多類別標籤）。
        #self.validate_labels() #checking if the labels are binary or continuous.

        if norm is not None:
            self.norm = norm #將 norm 存儲在 self.norm 中供後續使用。（如 'standard'、'minmax' 等）。
            
            if not self.check_binary_labels(y_train) and not self.check_multiclass_labels(y_train): #y_train is continuous確認y_train是連續的數據，進行正規化
                self.y_train = self.normalize_labels(self.y_train, self.norm)
                print(f" continuous data, {self.norm} normalized ")
                print(self.y_train)

            if not self.check_binary_labels(y_test) and not self.check_multiclass_labels(y_test): #y_test進行相同的處理
                self.y_test = self.normalize_labels(self.y_test, self.norm)
                print(f" continuous data, {self.norm} normalized ")
                print(self.y_test)

        #print(self.y_train)
        self.is_binary = self.check_binary_labels(y_train) and self.check_binary_labels(y_test) #確認y_train和y_test都是二元標籤，判斷後續處理

        self.is_multiclass = self.check_multiclass_labels(y_train) or self.check_multiclass_labels(y_test) #確認y_train和y_test其中一個是多標籤，判斷後續處理
        
        if self.is_multiclass:
            self.num_classes = max(len(np.unique(y_train)),len(np.unique(y_test))) #如為多類別，則計算y_train和y_test中唯一標籤的最大數量
        
        
        
        
        
        
        self.train_dataset = DocWindowsDataset(doc_windows)

        if doc_lens is None:
            self.docweights = np.ones(ndocs, dtype=np.float)
        else:
            self.docweights = 1.0 / np.log(doc_lens)
            self.doc_lens = doc_lens
        self.docweights = torch.tensor(self.docweights, dtype=torch.float32, requires_grad=False, device=device)

        if expvars_train is not None:
            self.expvars_train = torch.tensor(expvars_train, dtype=torch.float32, requires_grad=False, device=device)
        if expvars_test is not None:
            self.expvars_test = torch.tensor(expvars_test, dtype=torch.float32, requires_grad=False, device=device)
        # word embedding
        self.embedding_i = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=embed_dim,
                                        sparse=False)
        if word_vectors is not None:
            self.embedding_i.weight.data = torch.FloatTensor(word_vectors)
        else:
            torch.nn.init.kaiming_normal_(self.embedding_i.weight)

        # regular embedding for concepts (never indexed so not sparse)
        self.embedding_t = nn.Parameter(torch.FloatTensor(ortho_group.rvs(embed_dim)[0:nconcepts]))
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        if file_log:
            log_path = os.path.join(out_dir, "gdcm.log")
            print("Saving logs in the file " + os.path.abspath(log_path))
            logging.basicConfig(filename=log_path,
                                format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

        # embedding for per-document concept weights
        if self.inductive:
            weight_generator_network = []
            if num_layers > 0:
                # input layer
                weight_generator_network.extend([torch.nn.Linear(vocab_size, hidden_size),
                                                 torch.nn.Tanh(),
                                                 torch.nn.Dropout(inductive_dropout)])
                # hidden layers
                for h in range(num_layers):
                    weight_generator_network.extend([torch.nn.Linear(hidden_size, hidden_size),
                                                     torch.nn.Tanh(),
                                                     torch.nn.Dropout(inductive_dropout)])
                # output layer
                weight_generator_network.append(torch.nn.Linear(hidden_size,
                                                                nconcepts))
            else:
                weight_generator_network.append(torch.nn.Linear(vocab_size,
                                                                nconcepts))
            for m in weight_generator_network:
                if type(m) == torch.nn.Linear:
                    torch.nn.init.normal_(m.weight)
                    torch.nn.init.normal_(m.bias)
            self.doc_concept_network = torch.nn.Sequential(*weight_generator_network)
        else:
            self.doc_concept_weights = nn.Embedding(num_embeddings=ndocs,
                                                    embedding_dim=nconcepts,
                                                    sparse=False)
            if doc_concept_probs is not None:
                self.doc_concept_weights.weight.data = torch.FloatTensor(doc_concept_probs)
            else:
                torch.nn.init.kaiming_normal_(self.doc_concept_weights.weight)

        if theta is not None:
            self.theta = Parameter(torch.FloatTensor(theta))
        # explanatory variables
        else:
            if expvars_train is not None:
                # TODO: add assert shape
                nexpvars = expvars_train.shape[1]
                if self.is_multiclass:
                    self.theta = Parameter(torch.FloatTensor(nconcepts + nexpvars + 1, self.num_classes))  # + 1 for bias
                else:
                    self.theta = Parameter(torch.FloatTensor(nconcepts + nexpvars + 1))  # for binary and continuous variables
            
            
            else:
                if self.is_multiclass:
                    self.theta = Parameter(torch.FloatTensor(nconcepts + 1, self.num_classes))  # + 1 for bias
                else:
                    self.theta = Parameter(torch.FloatTensor(nconcepts + 1))  # for binary and continuous variables
        
        torch.nn.init.normal_(self.theta)

        # enable gradients (True by default, just confirming)
        self.embedding_i.weight.requires_grad = True
        self.embedding_t.requires_grad = True
        self.theta.requires_grad = True

        # weights for negative sampling
        wf = np.power(word_counts, consts.BETA)  # exponent from word2vec paper
        self.word_counts = word_counts
        wf = wf / np.sum(wf)  # convert to probabilities
        self.weights = torch.tensor(wf, dtype=torch.float32, requires_grad=False, device=device)
        self.vocab = vocab
        # dropout
        self.dropout1 = nn.Dropout(consts.PIVOTS_DROPOUT)
        self.dropout2 = nn.Dropout(consts.DOC_VECS_DROPOUT)
        self.multinomial = AliasMultinomial(wf, self.device)

    """ code optimization task 1 (for checking if y_train/y_test is binary or continuous in the existing datasets)"""