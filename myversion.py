import re
import os
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

#######################
# 1. 讀取並處理文本文件
#######################

# 假設 alt.atheism.txt 與此腳本在同一目錄下
file_path = "alt.atheism.txt"
with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# 根據雙換行分割成多個段落（作為獨立文檔）
docs = text.split("\n\n")
docs = [doc.strip() for doc in docs if doc.strip() != ""]

# 定義預處理函數：轉小寫、去除非字母字符、分詞
def preprocess(doc):
    doc = doc.lower()
    doc = re.sub(r"[^a-z\s]", "", doc)
    tokens = doc.split()
    return tokens

docs_tokens = [preprocess(doc) for doc in docs]

###################################
# 2. 建立詞彙表並生成詞袋表示
###################################

# 建立詞彙：這裡簡單採用所有出現過的單詞
all_tokens = [token for doc in docs_tokens for token in doc]
vocab_counter = Counter(all_tokens)
vocab = list(vocab_counter.keys())
vocab_size = len(vocab)
print(f"建立詞彙表，大小 = {vocab_size}")

# 建立單詞到索引的映射
word2idx = {word: i for i, word in enumerate(vocab)}

# 將每個文檔轉換為詞袋向量（每個向量大小為 vocab_size）
bow_matrix = []
for tokens in docs_tokens:
    vec = np.zeros(vocab_size, dtype=np.float32)
    for token in tokens:
        idx = word2idx[token]
        vec[idx] += 1
    bow_matrix.append(vec)
bow_matrix = np.array(bow_matrix)
num_docs = bow_matrix.shape[0]
print(f"文檔數量 = {num_docs}")

###################################
# 3. 構造 DataLoader
###################################

# 將 numpy 數據轉換為 torch tensor
bow_tensor = torch.tensor(bow_matrix, dtype=torch.float32)
# 使用 TensorDataset 和 DataLoader（這裡不使用標籤，僅用詞袋向量）
dataset = TensorDataset(bow_tensor)
batch_size = 16
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

###################################
# 4. 定義基於VAE的主題模型
###################################

class VAE_TopicModel(nn.Module):
    def __init__(self, vocab_size, n_topics, hidden_size=100):
        """
        :param vocab_size: 詞彙表大小
        :param n_topics: 主題數量（潛在變量維度）
        :param hidden_size: 隱藏層大小
        """
        super(VAE_TopicModel, self).__init__()
        self.vocab_size = vocab_size
        self.n_topics = n_topics
        self.hidden_size = hidden_size
        
        # 編碼器部分
        self.fc1 = nn.Linear(vocab_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, n_topics)
        self.fc_logvar = nn.Linear(hidden_size, n_topics)
        
        # 解碼器部分：將潛在主題轉換回詞袋分布
        self.fc_decoder = nn.Linear(n_topics, vocab_size)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        logits = self.fc_decoder(z)
        # 使用 softmax 將 logits 轉換成概率分布（每個文檔中各詞的出現概率）
        return F.softmax(logits, dim=1)
    
    def forward(self, x):
        # x 的形狀為 [batch_size, vocab_size]（詞袋向量）
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def loss_function(recon, x, mu, logvar):
    """
    定義損失：重構損失（交叉熵形式）+ KL 散度
    """
    # 重構損失：計算原始詞袋與重構分布之間的交叉熵（對每個文檔求和）
    BCE = -torch.sum(x * torch.log(recon + 1e-10), dim=1)
    # KL 散度：讓潛在變量接近標準正態分布
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return torch.mean(BCE + KLD)

###################################
# 5. 訓練模型
###################################

def train_topic_model(model, data_loader, epochs=50, lr=1e-3, device="cpu"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in data_loader:
            # TensorDataset 返回的是一個 tuple，取出第一個元素即詞袋向量
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = loss_function(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 設定主題模型參數：這裡設定主題數量為 20
n_topics = 20
model = VAE_TopicModel(vocab_size, n_topics, hidden_size=100).to(device)

# 開始訓練（訓練輪數可根據需要調整）
train_topic_model(model, data_loader, epochs=50, lr=1e-3, device=device)

###################################
# 6. 查看每個主題的重要詞語（透過解碼器權重）
###################################
# 取出解碼器的權重，權重 shape 為 [n_topics, vocab_size]
phi_logits = model.fc_decoder.weight.data.cpu().numpy()
# 對每個主題取出排名前 10 的詞語
top_n = 10
for topic_idx, topic_weights in enumerate(phi_logits):
    top_word_indices = topic_weights.argsort()[-top_n:][::-1]
    top_words = [vocab[i] for i in top_word_indices]
    print(f"Topic {topic_idx+1}: {' '.join(top_words)}")
