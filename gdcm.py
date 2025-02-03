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
        

        
        
        
        
        self.train_dataset = DocWindowsDataset(doc_windows) #把有關上下文的資料丟入，建立一個DocWindowsDataset的物件

        if doc_lens is None: #doc_lens : 每個文件的長度(詞彙數量)，形狀為(n_train_docs,)。
            self.docweights = np.ones(ndocs, dtype=np.float) #ndocs : 文件的總數，即bow_train.shape[0]。 self.docweights : 儲存計算出的文件權重，後續可能用於調整損失函數或模型訓練權重。
        else:
            self.docweights = 1.0 / np.log(doc_lens)
            self.doc_lens = doc_lens
        self.docweights = torch.tensor(self.docweights, dtype=torch.float32, requires_grad=False, device=device)
        """檢查doc_lens是否為空，如果為空(沒有提供文件長度資訊)，將docweights全設定為1的陣列，不考慮文件長度的影響。
           有提供doc_lens的話，計算文件的權重，較長文件權重低，較短文件權重高，類似TF-IDF的標準化技術
           最後轉為PyTorch的張量(tensor)，以便使用GPU加速運算和PyTorch模型相容。
           requires_grad=False：表示這個張量不需要梯度更新，通常用於靜態權重或參數。"""

        if expvars_train is not None: #如果有額外的變數，把其也轉成PyTorch張量
            self.expvars_train = torch.tensor(expvars_train, dtype=torch.float32, requires_grad=False, device=device)
        if expvars_test is not None:
            self.expvars_test = torch.tensor(expvars_test, dtype=torch.float32, requires_grad=False, device=device)
        # word embedding
        self.embedding_i = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=embed_dim,
                                        sparse=False)
        '''nn.Embedding 是 PyTorch 提供的嵌入層，可以將離散的單詞索引（如詞彙 ID）轉換為連續的向量（詞向量）。
           num_embeddings=vocab_size：詞彙表大小（vocab_size），也就是模型能夠學習的不同詞的數量。
           embedding_dim=embed_dim：詞向量的維度，通常設為 100、200 或 300（根據應用場景）。
           sparse=False：
             是否使用稀疏梯度更新：
               True：適用於大詞彙表，節省內存，但僅適用於 SGD 優化器。
               False：適用於大部分優化器（如 Adam），會更新所有權重。'''
        if word_vectors is not None:
            self.embedding_i.weight.data = torch.FloatTensor(word_vectors) #如果有預先訓練好的詞向量 (word_vectors)，則直接將其賦值給嵌入層的權重 (self.embedding_i.weight.data)。
        else:
            torch.nn.init.kaiming_normal_(self.embedding_i.weight) #如果沒有提供預先訓練的詞向量，則使用 Kaiming 正規初始化 (kaiming_normal_) 來初始化詞向量的權重

        # regular embedding for concepts (never indexed so not sparse)
        self.embedding_t = nn.Parameter(torch.FloatTensor(ortho_group.rvs(embed_dim)[0:nconcepts]))
        '''
            self.embedding_t：這是用來表示「概念」的嵌入向量 (concept embeddings)。
            nn.Parameter(...)：
            讓這個變數成為可學習參數，它會在訓練過程中自動更新。
            這與 nn.Embedding 不同，因為 nn.Embedding 是用來查找離散索引，而這裡是直接定義一組可學習的概念向量。
            ortho_group.rvs(embed_dim)[0:nconcepts]：
            這個函數來自 scipy.stats，用來生成隨機正交矩陣（Orthogonal Matrix）。
            正交矩陣 可以確保概念向量之間的距離盡可能均勻分佈，減少相關性 (避免過度擬合)。
            rvs(embed_dim) 生成一個 embed_dim × embed_dim 的矩陣，我們取前 nconcepts 行來初始化概念向量。
            為什麼使用正交矩陣來初始化？
            確保概念向量之間的多樣性，避免過度相似。
            提升數值穩定性，防止梯度消失或梯度爆炸。
            促進學習不同的語意概念，適合主題建模 (Topic Modeling) 或語意分析任務。'''
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
        '''
            刪除舊的日誌處理器，避免重複輸出。
            支援將日誌寫入檔案或輸出到終端，方便後續調試。
            設定日誌格式，讓輸出更具可讀性。
            建立 self.logger 物件，後續可以用 self.logger.debug(...) 來記錄資訊。'''

        # embedding for per-document concept weights
        '''
        如果self.inductive==True，則模型不直接儲存概念權重。
        使用神經網路，由BOW向量產生概念權重。
        Appendix A.3'''
        if self.inductive:
            weight_generator_network = [] #用來存放神經網路各層(linear, tanh, dropout)的list
            if num_layers > 0:
                # input layer
                weight_generator_network.extend([torch.nn.Linear(vocab_size, hidden_size), #將BOW向量轉為隱藏層
                                                 torch.nn.Tanh(), #激活函數，讓數值落在[-1, 1]
                                                 torch.nn.Dropout(inductive_dropout)]) #隨機丟棄一些神經元，防止overfitting
                # hidden layers
                for h in range(num_layers): #num_layers個隱藏層，每個維度都一樣
                    weight_generator_network.extend([torch.nn.Linear(hidden_size, hidden_size),
                                                     torch.nn.Tanh(),
                                                     torch.nn.Dropout(inductive_dropout)])
                # output layer
                weight_generator_network.append(torch.nn.Linear(hidden_size, #隱藏層映射到概念的數量
                                                                nconcepts))
            else:
                weight_generator_network.append(torch.nn.Linear(vocab_size,
                                                                nconcepts))
            for m in weight_generator_network: #確保所有Linear層的權重(weight)和偏差(bias)被初始化成服從常態分布，避免梯度消失或爆炸
                if type(m) == torch.nn.Linear:
                    torch.nn.init.normal_(m.weight)
                    torch.nn.init.normal_(m.bias)
            self.doc_concept_network = torch.nn.Sequential(*weight_generator_network) #將所有Linear、Tanh和Dropout層組成一個Sequential神經網路
        else:
            self.doc_concept_weights = nn.Embedding(num_embeddings=ndocs,
                                                    embedding_dim=nconcepts,
                                                    sparse=False)
            '''如果 self.inductive == False，則 直接用 nn.Embedding 來存每個文件的概念權重，不使用神經網路。
            ndocs：文件數量，每個文件都有一組概念權重。
            nconcepts：每個文件的概念數量。
            sparse=False：不啟用稀疏更新，讓梯度能夠更新所有權重。'''
            
            if doc_concept_probs is not None:
                self.doc_concept_weights.weight.data = torch.FloatTensor(doc_concept_probs) # 如果有預設概念權重，則載入
            else:
                torch.nn.init.kaiming_normal_(self.doc_concept_weights.weight) #如果沒有預設權重，則使用 Kaiming 初始化

        if theta is not None:
            self.theta = Parameter(torch.FloatTensor(theta)) #如果 theta 不是 None，則將其轉換成 torch.FloatTensor，並包裝成 nn.Parameter。
        # explanatory variables 以下作用為初始化theta
        else:
            if expvars_train is not None:
                # TODO: add assert shape
                nexpvars = expvars_train.shape[1] #得到可解釋變數的數量
                if self.is_multiclass: #決定theta的形狀
                    self.theta = Parameter(torch.FloatTensor(nconcepts + nexpvars + 1, self.num_classes))  # + 1 for bias
                else:
                    self.theta = Parameter(torch.FloatTensor(nconcepts + nexpvars + 1))  # for binary and continuous variables
            
            
            else:
                if self.is_multiclass:
                    self.theta = Parameter(torch.FloatTensor(nconcepts + 1, self.num_classes))  # + 1 for bias
                else:
                    self.theta = Parameter(torch.FloatTensor(nconcepts + 1))  # for binary and continuous variables
        
        torch.nn.init.normal_(self.theta) #使用常態分布初始化theta，讓模型在初始時不會有過大的數值，避免梯度爆炸或消失的問題。

        # enable gradients (True by default, just confirming) 確保embeddinng_i, embedding_t, theta都能參與梯度更新
        self.embedding_i.weight.requires_grad = True
        self.embedding_t.requires_grad = True
        self.theta.requires_grad = True

        # weights for negative sampling
        wf = np.power(word_counts, consts.BETA)  # exponent from word2vec paper
        self.word_counts = word_counts
        wf = wf / np.sum(wf)  # convert to probabilities
        self.weights = torch.tensor(wf, dtype=torch.float32, requires_grad=False, device=device)
        self.vocab = vocab
        '''
        這部分與 Word2Vec 的負樣本抽樣方法類似。
        word_counts：單詞出現的次數。
        consts.BETA：控制對高頻單詞的懲罰 (常見值為 0.75)。
        這個公式讓常見的單詞被降低權重，而較罕見的單詞增加機率。
        最後轉成 torch.tensor，並設置 requires_grad=False，確保它不會被訓練更新。'''
        # dropout
        self.dropout1 = nn.Dropout(consts.PIVOTS_DROPOUT)
        self.dropout2 = nn.Dropout(consts.DOC_VECS_DROPOUT)
        '''self.dropout1 和 self.dropout2 負責在不同的部分隨機丟棄神經元，以防止過擬合：
                consts.PIVOTS_DROPOUT：用於樞軸詞 (pivot words)。
                consts.DOC_VECS_DROPOUT：用於文件向量 (document vectors)。'''
        self.multinomial = AliasMultinomial(wf, self.device)
        '''AliasMultinomial 是一種高效的 非均勻隨機抽樣方法。
        這裡使用 wf (負樣本權重) 來進行 負樣本抽樣 (Negative Sampling)，類似於 Word2Vec 的技巧。
        '''

    """ code optimization task 1 (for checking if y_train/y_test is binary or continuous in the existing datasets)"""

def check_binary_labels(self, y): #檢查y是否為二元標籤
    unique_values = np.unique(y)
    
    return (len(unique_values) == 2 and set(unique_values).issubset({0, 1}))


def check_multiclass_labels(self, y): #檢查y使否為多類別，且不是數值資料。ex : ['cat', 'dog', 'fish']
    unique_values = np.unique(y)
    num_of_vals = len(unique_values)
    return num_of_vals > 2 and not np.issubdtype(y.dtype, np.number)


def validate_labels(self): #驗證y_train和y_test是否為二元分類
    if self.check_binary_labels(self.y_train):
        print("y_train is binary")
    else:
        print("y_train is continuous, normalizing y_train...")
    if self.check_binary_labels(self.y_test):
        print("y_test is binary")
    else:
        print("y_test is continuous, normalizing y_test...")

def normalize_labels(self, data, method='standard'): #對y_train或y_test進行標準化
    if method == 'standard':
        scaler = StandardScaler() #使用Z-score標準化
    elif method == 'minmax':
        scaler = MinMaxScaler() #使用Min-Max標準化
    elif method == 'robust':
        scaler = RobustScaler() #使用Robust標準化
    else:
        raise ValueError("Unsupported normalization method")
    
    return scaler.fit_transform(data.reshape(-1, 1)).flatten() #將data轉為2D矩陣，因為scaler.fit_transform()需要2D輸入。使用flatten()轉回1D，以維持原始y的形狀。




def forward(self, doc, target, contexts, labels, per_doc_loss=None):
    """
    Args:
        doc:        [batch_size,1] LongTensor of document indices
        target:     [batch_size,1] LongTensor of target (pivot) word indices
        contexts:   [batchsize,window_size] LongTensor of convext word indices
        labels:     [batchsize,1] LongTensor of document labels

        All arguments are tensors wrapped in Variables.

        此函式的主要用途是計算LDA2Vec的loss function
        1. 負採樣損失 (Negative Sampling Loss) : 用於詞嵌入學習
        2. Dirichlet Loss : 用於主題建模 (Topic Modeling)
        3. 預測損失 (Prediction Loss) : 用於預測labels
        4. 多樣性損失 (Diversity Loss) : 用於鼓勵不同的概念向量(Concept Vectors)具有較大差異
    """
    batch_size, window_size = contexts.size()

    # reweight loss by document length
    w = autograd.Variable(self.docweights[doc.data]).to(self.device) #將tensor包進Variable中，支援backward()。autograd(自動微分)
    w /= w.sum() #計算文件的權重
    w *= w.size(0) #w會依據文檔長度進行標準化，以確保長文檔不會過度影響模型。

    # construct document vector = weighted linear combination of concept vectors
    if self.inductive:
        doc_concept_weights = self.doc_concept_network(self.bow_train[doc]) #在CAN計算出來的
    else:
        doc_concept_weights = self.doc_concept_weights(doc) #同樣是CAN
    doc_concept_probs = F.softmax(doc_concept_weights, dim=1)
    '''
        文檔向量是由概念向量 (concept vectors) 加權組合而成
        透過softmax計算主題機率 (doc_concept_probs)，代表每個文檔與不同概念的關聯程度
    '''
    doc_concept_probs = doc_concept_probs.unsqueeze(1)  # (batches, 1, T)
    concept_embeddings = self.embedding_t.expand(batch_size, -1, -1)  # (batches, T, E)
    doc_vector = torch.bmm(doc_concept_probs, concept_embeddings)  # (batches, 1, E)
    doc_vector = doc_vector.squeeze(dim=1)  # (batches, E)
    doc_vector = self.dropout2(doc_vector)
    """
        使用批次矩陣乘法(torch.bmm)，將主題機率(doc_concept_probs)與主題嵌入(concept embeddings)相乘，得到文檔向量(doc_vector)
    """

    # sample negative word indices for negative sampling loss; approximation by sampling from the whole vocab
    if self.device == "cpu":
        nwords = torch.multinomial(self.weights, batch_size * window_size * self.nnegs,
                                    replacement=True).view(batch_size, -1)
        nwords = autograd.Variable(nwords)
    else:
        nwords = self.multinomial.draw(batch_size * window_size * self.nnegs)
        nwords = autograd.Variable(nwords).view(batch_size, window_size * self.nnegs)
    """
        負採樣透過torch.multinomial從整個詞彙表(vocabulary)中隨機抽取nnegs個詞作為負樣本(negative samples)
        用來近似word2vec的Skip-gram模型，以提高訓練效率
    """

    # compute word vectors
    ivectors = self.dropout1(self.embedding_i(target))  # (batches, E)
    ovectors = self.embedding_i(contexts)  # (batches, window_size, E)
    nvectors = self.embedding_i(nwords).neg()  # row vector
    """
        計算目標詞和負樣本詞向量
        ivectors : 目標詞向量
        ovectors : 上下文詞向量
        nvectors : 負樣本詞向量
    """

    # construct "context" vector defined by lda2vec
    context_vectors = doc_vector + ivectors
    context_vectors = context_vectors.unsqueeze(2)  # (batches, E, 1)
    """
        context_vectors = 文檔向量 + 目標詞向量
        context_vectors表示LDA2Vec定義的語境 (Context Representation)
    """

    # compose negative sampling loss
    oloss = torch.bmm(ovectors, context_vectors).squeeze(dim=2).sigmoid().clamp(min=consts.EPS).log().sum(1)
    nloss = torch.bmm(nvectors, context_vectors).squeeze(dim=2).sigmoid().clamp(min=consts.EPS).log().sum(1)
    negative_sampling_loss = (oloss + nloss).neg()
    negative_sampling_loss *= w  # downweight loss for each document
    negative_sampling_loss = negative_sampling_loss.mean()  # mean over the batch
    """
        正樣本損失nloss : 計算ovectors和context_vectors之間的內積，並通過sigmoid運算
        負樣本損失oloss : 計算nvectors和context_vectors之間的內積
        最終損失negative_sampling_loss : 兩者相加取負號，然後根據文檔長度w加權
        公式4
    """

    # compose dirichlet loss
    doc_concept_probs = doc_concept_probs.squeeze(dim=1)  # (batches, T)
    doc_concept_probs = doc_concept_probs.clamp(min=consts.EPS)
    dirichlet_loss = doc_concept_probs.log().sum(1)  # (batches, 1)
    dirichlet_loss *= self.lam * (1.0 - self.alpha)
    dirichlet_loss *= w  # downweight loss for each document
    dirichlet_loss = dirichlet_loss.mean()  # mean over the entire batch
    """
        Dirichlet Loss控制主題分布，使的不同文檔的主題權重更加平滑
        self.lam和self.alpha是正則化超參數
        公式5
    """

    ones = torch.ones((batch_size, 1)).to(self.device)
    doc_concept_probs = torch.cat((ones, doc_concept_probs), dim=1)

    # expand doc_concept_probs vector with explanatory variables
    if self.expvars_train is not None:
        doc_concept_probs = torch.cat((doc_concept_probs, self.expvars_train[doc, :]),
                                        dim=1)
    # compose prediction loss
    # [batch_size] = torch.matmul([batch_size, nconcepts], [nconcepts])
    # pred_weight = torch.matmul(doc_concept_probs.unsqueeze(0), self.theta).squeeze(0)
    # print(doc_concept_probs.shape)
    # print(self.theta.shape)
    pred_weight = torch.matmul(doc_concept_probs, self.theta)
    # print(pred_weight)
    # print(labels)

    if self.is_binary:
        pred_loss = F.binary_cross_entropy_with_logits(pred_weight, labels,
                                                    weight=w, reduction='none')
    elif self.is_multiclass:
        pred_loss = F.cross_entropy(pred_weight, labels.long(), reduction = 'none')
    else:
        pred_loss = F.mse_loss(pred_weight, labels, reduction='none')
        pred_loss = pred_loss * w # applying the weight element-wise to the calcuated loss

    pred_loss *= self.rho
    pred_loss = pred_loss.mean()
    """
        計算預測損失 (Prediction Loss)
        binary_cross_entropy_with_logits適用於二元分類
        cross_entropy適用於多分類
        mse_loss適用於回歸
        公式8
    """

    # compose diversity loss
    #   1. First compute \sum_i \sum_j log(sigmoid(T_i, T_j))
    #   2. Then compute \sum_i log(sigmoid(T_i, T_i))
    #   3. Loss = (\sum_i \sum_j log(sigmoid(T_i, T_j)) - \sum_i log(sigmoid(T_i, T_i)) )
    #           = \sum_i \sum_{j > i} log(sigmoid(T_i, T_j))
    div_loss = torch.mm(self.embedding_t,
                        torch.t(self.embedding_t)).sigmoid().clamp(min=consts.EPS).log().sum() \
                - (self.embedding_t * self.embedding_t).sigmoid().clamp(min=consts.EPS).log().sum()
    div_loss /= 2.0  # taking care of duplicate pairs T_i, T_j and T_j, T_i
    div_loss = div_loss.repeat(batch_size)
    div_loss *= w  # downweight by document lengths
    div_loss *= self.eta
    div_loss = div_loss.mean()  # mean over the entire batch
    """
        計算多樣性損失 (Diversity Loss)
        確保不同的主題嵌入 (Topic Embeddings) 之間盡可能不相似，避免過度重疊
        公式6
    """

    return negative_sampling_loss, dirichlet_loss, pred_loss, div_loss
    #返回4種損失，作為模型學習的目標

def fit(self, lr=0.01, nepochs=200, pred_only_epochs=20,
        batch_size=100, weight_decay=0.01, grad_clip=5, save_epochs=10, concept_dist="dot"):
    """
    Train the GDCM model

    Parameters
    ----------
    lr : float
        Learning rate 學習率
    nepochs : int
        The number of training epochs 訓練的總epoch數
    pred_only_epochs : int
        The number of epochs optimized with prediction loss only 只優化預測損失(prediction loss)的epochs數量
    batch_size : int
        Batch size 批次大小
    weight_decay : float
        Adam optimizer weight decay (L2 penalty) Adam優化器的L2正則化權重衰減
    grad_clip : float
        Maximum gradients magnitude. Gradients will be clipped within the range [-grad_clip, grad_clip] 梯度裁減的範圍，避免梯度爆炸
    save_epochs : int
        The number of epochs in between saving the model weights 每saved_epochs個epoch儲存一次模型
    concept_dist: str
        Concept vectors distance metric. Choices are 'dot', 'correlation', 'cosine', 'euclidean', 'hamming'. 概念向量的距離度量(可選擇dot, correlation, cosine, euclidean, hamming)

    Returns
    -------
    metrics : ndarray, shape (n_epochs, 6)
        Training metrics from each epoch including: total_loss, avg_sgns_loss, avg_diversity_loss, avg_pred_loss,
        avg_diversity_loss, train_auc, test_auc
    -------
    這個fit方法是GDCM模型的訓練主函式
    1. 讀取訓練數據並設定Adam優化器
    2. 進行nepochs次的訓練，每個batch計算SGNS、Dirichlet、預測、多樣性損失
    3. 根據不同的epoch階段，調整loss function以確保模型逐步收斂
    4. 使用梯度裁減(grad_clip)防止梯度爆炸
    5. 計算並記錄AUC，每save_epochs儲存一次模型
    6. 訓練完成後儲存最終模型並回傳訓練結果
    """
    train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size, shuffle=True,
                                                    num_workers=4, pin_memory=True,
                                                    drop_last=False)

    
    
    self.to(self.device)

    train_metrics_file = open(os.path.join(self.out_dir, "train_metrics.txt"), "w")
    train_metrics_file.write("total_loss,avg_sgns_loss,avg_dirichlet_loss,avg_pred_loss,"
                                "avg_div_loss,train_auc,test_auc\n")

    # SGD generalizes better: https://arxiv.org/abs/1705.08292
    optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
    nwindows = len(self.train_dataset)
    results = []
    for epoch in range(nepochs):
        total_sgns_loss = 0.0
        total_dirichlet_loss = 0.0
        total_pred_loss = 0.0
        total_diversity_loss = 0.0

        self.train()
        for batch in train_dataloader:
            batch = batch.long()
            batch = batch.to(self.device)
            doc = batch[:, 0]
            iword = batch[:, 1]
            owords = batch[:, 2:-1]
            labels = batch[:, -1].float()

            sgns_loss, dirichlet_loss, pred_loss, div_loss = self(doc, iword, owords, labels)
            if epoch < pred_only_epochs:
                loss = pred_loss
            else:
                loss = sgns_loss + dirichlet_loss + pred_loss + div_loss
            optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            for p in self.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad = p.grad.clamp(min=-grad_clip, max=grad_clip)

            optimizer.step()

            nsamples = batch.size(0)

            total_sgns_loss += sgns_loss.detach().cpu().numpy() * nsamples
            total_dirichlet_loss += dirichlet_loss.detach().cpu().numpy() * nsamples
            total_pred_loss += pred_loss.data.detach().cpu().numpy() * nsamples
            total_diversity_loss += div_loss.data.detach().cpu().numpy() * nsamples

        train_auc = self.calculate_auc("Train", self.bow_train, self.y_train, self.expvars_train)
        test_auc = 0.0
        if self.inductive:
            test_auc = self.calculate_auc("Test", self.bow_test, self.y_test, self.expvars_test)

        total_loss = (total_sgns_loss + total_dirichlet_loss + total_pred_loss + total_diversity_loss) / nwindows
        avg_sgns_loss = total_sgns_loss / nwindows
        avg_dirichlet_loss = total_dirichlet_loss / nwindows
        avg_pred_loss = total_pred_loss / nwindows
        avg_diversity_loss = total_diversity_loss / nwindows
        self.logger.info("epoch %d/%d:" % (epoch, nepochs))
        self.logger.info("Total loss: %.4f" % total_loss)
        self.logger.info("SGNS loss: %.4f" % avg_sgns_loss)
        self.logger.info("Dirichlet loss: %.4f" % avg_dirichlet_loss)
        self.logger.info("Prediction loss: %.4f" % avg_pred_loss)
        self.logger.info("Diversity loss: %.4f" % avg_diversity_loss)
        concepts = self.get_concept_words(concept_dist=concept_dist)
        with open(os.path.join(self.concept_dir, "epoch%d.txt" % epoch), "w") as concept_file:
            for i, concept_words in enumerate(concepts):
                self.logger.info('concept %d: %s' % (i + 1, ' '.join(concept_words)))
                concept_file.write('concept %d: %s\n' % (i + 1, ' '.join(concept_words)))
        metrics = (total_loss, avg_sgns_loss, avg_dirichlet_loss, avg_pred_loss,
                    avg_diversity_loss, train_auc, test_auc)
        train_metrics_file.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" % metrics)
        train_metrics_file.flush()
        results.append(metrics)
        if (epoch + 1) % save_epochs == 0:
            torch.save(self.state_dict(), os.path.join(self.model_dir, "epoch%d.pytorch" % epoch))
            with torch.no_grad():
                doc_concept_probs = self.get_train_doc_concept_probs()
                np.save(os.path.join(self.model_dir, "epoch%d_train_doc_concept_probs.npy" % epoch),
                        doc_concept_probs.cpu().detach().numpy())

    torch.save(self.state_dict(), os.path.join(self.model_dir, "epoch%d.pytorch" % (nepochs - 1)))
    return np.array(results)

def calculate_auc(self, split, X, y, expvars):
    y_pred = self.predict_proba(X, expvars).cpu().detach().numpy()
    if self.is_binary:
        auc = roc_auc_score(y, y_pred)
        self.logger.info("%s AUC: %.4f" % (split, auc))
        return auc
    elif self.is_multiclass:
        y = np.asarray(y).astype(int)
    
        num_classes = len(np.unique(y))
        
        if num_classes > 1:  
            
            y_pred_normalized = y_pred / y_pred.sum(axis=1, keepdims=True)
            auc = roc_auc_score(y, y_pred_normalized, multi_class='ovr')
            self.logger.info("%s AUC (OvR): %.4f" % (split, auc))
            return auc
        else:
            #in case only one class is present
            self.logger.warning("Only one class present in true labels. ROC AUC score is not defined in that case.")
            return None  
    else:
        mse = mean_squared_error(y, y_pred)
        # mae = mean_absolute_error(y, y_pred)
        # r2 = r2_score(y, y_pred)
        self.logger.info("%s MSE: %.4f" % (split, mse))
        return mse
        

def predict_proba(self, count_matrix, expvars=None):
    with torch.no_grad():
        batch_size = count_matrix.size(0)
        if self.inductive:
            doc_concept_weights = self.doc_concept_network(count_matrix)
        else:
            doc_concept_weights = self.doc_concept_weights.weight.data
        doc_concept_probs = F.softmax(doc_concept_weights, dim=1)  # convert to probabilities
        ones = torch.ones((batch_size, 1)).to(self.device)
        doc_concept_probs = torch.cat((ones, doc_concept_probs), dim=1)

        if expvars is not None:
            doc_concept_probs = torch.cat((doc_concept_probs, expvars), dim=1)

        pred_weight = torch.matmul(doc_concept_probs, self.theta)
        pred_proba = pred_weight.sigmoid()
    return pred_proba

def get_train_doc_concept_probs(self):
    if self.inductive:
        doc_concept_weights = self.doc_concept_network(self.bow_train)
    else:
        doc_concept_weights = self.doc_concept_weights.weight.data
    return F.softmax(doc_concept_weights, dim=1)  # convert to probabilities

def visualize(self):
    with torch.no_grad():
        doc_concept_probs = self.get_train_doc_concept_probs()
        # [n_concepts, vocab_size] weighted word counts of each concept
        concept_word_counts = torch.matmul(doc_concept_probs.transpose(0, 1), self.bow_train)
        # normalize word counts to word distribution of each concept
        concept_word_dists = concept_word_counts / concept_word_counts.sum(1, True)
        # fill NaN with 1/vocab_size in case a concept has all zero word distribution
        concept_word_dists[concept_word_dists != concept_word_dists] = 1.0 / concept_word_dists.shape[1]
        vis_data = pyLDAvis.prepare(topic_term_dists=concept_word_dists.data.cpu().numpy(),
                                    doc_topic_dists=doc_concept_probs.data.cpu().numpy(),
                                    doc_lengths=self.doc_lens, vocab=self.vocab, term_frequency=self.word_counts)
        
        html_path = os.path.join(self.out_dir, "visualization.html")
        pyLDAvis.save_html(vis_data, html_path)
        # pyLDAvis.save_html(vis_data, os.path.join(self.out_dir, "visualization.html"))

        for i in range(len(concept_word_dists)):
            concept_word_weights = dict(zip(self.vocab, concept_word_dists[i].cpu().numpy()))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(concept_word_weights)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f'Concept {i+1} Word Cloud')
            plt.axis('off')
            plt.savefig(os.path.join(self.out_dir, f'concept_{i+1}_wordcloud.png'))

        with open(html_path, "a+") as f:

            current_directory = os.getcwd()
            print("Current directory:", current_directory)
            swiper_vis_path = current_directory + '/swiper.html'
            with open(swiper_vis_path, 'r') as swipertext:
                swiper = swipertext.read() 
                
            f.write(swiper)
        

# TODO: add filtering such as pos and tf
def get_concept_words(self, top_k=10, concept_dist='dot'):
    concept_embed = self.embedding_t.data.cpu().numpy()
    word_embed = self.embedding_i.weight.data.cpu().numpy()
    if concept_dist == 'dot':
        dist = -np.matmul(concept_embed, np.transpose(word_embed, (1, 0)))
    else:
        dist = cdist(concept_embed, word_embed, metric=concept_dist)
    nearest_word_idxs = np.argsort(dist, axis=1)[:, :top_k]  # indices of words with min cosine distance
    concepts = []
    for j in range(self.nconcepts):
        nearest_words = [self.vocab[i] for i in nearest_word_idxs[j, :]]
        concepts.append(nearest_words)
    return concepts


class DocWindowsDataset(torch.utils.data.Dataset):
    def __init__(self, windows):
        self.data = windows

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
"""
此類別繼承自PyTorch的torch.utils.data.Dataset
提供了一個有效的介面來有效地處理和操作上下文視窗(或滑動視窗)資料，通常用於訓練或測試機器學習模型。
1. 目的
此類別封裝了一個由 windows 表示的資料集，其中每個「視窗」都是一列結構化的數據（例如包含文件索引、目標詞、上下文詞和標籤等）。這種視窗化的資料通常應用於自然語言處理（NLP）相關任務，例如從語料庫中生成上下文視窗，用於訓練或測試模型。

透過 PyTorch 的 Dataset API，這類數據可以高效地載入和處理，尤其是在搭配 DataLoader 時。

2. 建構函式：__init__
參數：
windows：類似陣列的物件（例如 numpy.ndarray 或 torch.Tensor），包含上下文視窗的資料集。
功能：
將輸入的 windows 儲存到實例變數 self.data 中，作為資料集的內部表示。

3. 長度：__len__
回傳值：
資料集中項目的數量（行數），即上下文視窗的總數。
功能：
支援與 PyTorch 的 DataLoader 相容，DataLoader 會使用此方法來決定要生成多少批次（batches）。

4. 取項目：__getitem__
參數：
idx：一個整數索引，指定要檢索的數據項目。
回傳值：
self.data 中第 idx 行的資料，表示單一的上下文視窗。
功能：
提供一種方法，透過索引訪問資料集中的單一項目，這對於訓練或推論時的批次處理非常重要。

為何使用這個類別？
自訂的 PyTorch 資料集：
繼承自 torch.utils.data.Dataset，確保與 PyTorch 的 DataLoader 相容，可自動化批次載入、隨機打亂和多線程處理。
處理上下文視窗：
此資料集專為儲存和檢索上下文視窗而設計，常見於 Word2Vec、主題建模或其他 NLP 任務。
可擴展性：
透過實作 __len__ 和 __getitem__ 方法，使此資料集能夠高效處理大規模數據。

"""