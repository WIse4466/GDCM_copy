import re
import os
import numpy as np
from collections import Counter
from sklearn.datasets import fetch_20newsgroups

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
docs = newsgroups.data
def preprocess(doc):
    doc = doc.lower()
    doc = re.sub(r"[^a-z\s]", "", doc)
    tokens = doc.split()
    return tokens

docs_tokens = [preprocess(doc) for doc in docs]
# 建立詞彙表
all_tokens = [token for doc in docs_tokens for token in doc]
vocab_counter = Counter(all_tokens)
vocab = list(vocab_counter.keys())
vocab_size = len(vocab)

# 建立 word2idx 和 idx2word 對應
word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for word, i in word2idx.items()}

print(f"詞彙表大小: {vocab_size}")

import random

def generate_skipgram_pairs(docs_tokens, window_size=2, num_neg_samples=5):
    positive_pairs = []
    negative_pairs = []
    
    for tokens in docs_tokens:
        indices = [word2idx[word] for word in tokens if word in word2idx]
        
        for i, target in enumerate(indices):
            window_start = max(i - window_size, 0)
            window_end = min(i + window_size + 1, len(indices))
            
            # 取得 context 單詞
            context_words = indices[window_start:i] + indices[i+1:window_end]
            for ctx in context_words:
                positive_pairs.append((target, ctx))
                
                # 生成負樣本
                for _ in range(num_neg_samples):
                    neg_word = random.randint(0, vocab_size - 1)
                    negative_pairs.append((target, neg_word))
    
    return positive_pairs, negative_pairs

# 產生 skip-gram 訓練數據
positive_pairs, negative_pairs = generate_skipgram_pairs(docs_tokens, window_size=2, num_neg_samples=5)
print(f"正樣本數量: {len(positive_pairs)}, 負樣本數量: {len(negative_pairs)}")

# 轉換成 PyTorch 張量
positive_pairs_tensor = torch.tensor(positive_pairs, dtype=torch.long)
negative_pairs_tensor = torch.tensor(negative_pairs, dtype=torch.long)

# 使用 TensorDataset 和 DataLoader
batch_size = 1024
# 正樣本 DataLoader
positive_dataset = TensorDataset(positive_pairs_tensor)
positive_data_loader = DataLoader(positive_dataset, batch_size=batch_size, shuffle=True)

# 負樣本 DataLoader
negative_dataset = TensorDataset(negative_pairs_tensor)
negative_data_loader = DataLoader(negative_dataset, batch_size=batch_size, shuffle=True)

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, center_word, context_word):
        v_pvt = self.embedding(center_word)  # 主要單詞向量
        v_ctx = self.embedding(context_word)  # 背景單詞向量
        return torch.sum(v_pvt * v_ctx, dim=1)  # 內積計算相似度

def negative_sampling_loss(pos_scores, neg_scores):
    """
    pos_scores: 正樣本的點積結果
    neg_scores: 負樣本的點積結果
    """
    pos_loss = -F.logsigmoid(pos_scores).mean()  # 第一項
    neg_loss = -F.logsigmoid(-neg_scores).mean()  # 第二項
    return pos_loss + neg_loss

# 設定超參數
embedding_dim = 100
learning_rate = 0.01
num_epochs = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkipGramModel(vocab_size, embedding_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 訓練迴圈
for epoch in range(num_epochs):
    total_loss = 0.0

    # 使用 zip() 同時迭代兩個 DataLoader
    for (pos_batch,), (neg_batch,) in zip(positive_data_loader, negative_data_loader):
        pos_batch, neg_batch = pos_batch.to(device), neg_batch.to(device)
        center_word_pos, context_word_pos = pos_batch[:, 0], pos_batch[:, 1]
        center_word_neg, context_word_neg = neg_batch[:, 0], neg_batch[:, 1]

        # 正樣本分數
        pos_scores = model(center_word_pos, context_word_pos)

        # 負樣本分數
        neg_scores = model(center_word_neg, context_word_neg)

        # 計算損失
        loss = negative_sampling_loss(pos_scores, neg_scores)

        # 反向傳播與優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

print("訓練完成！")

# 取得單詞的嵌入向量
word_embeddings = model.embedding.weight.data.cpu().numpy()

# 隨機選 10 個單詞，顯示其對應的嵌入向量
sample_words = random.sample(vocab, 10)
for word in sample_words:
    idx = word2idx[word]
    print(f"{word}: {word_embeddings[idx][:5]} ...")  # 顯示前 5 個數值