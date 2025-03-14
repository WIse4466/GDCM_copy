import numpy as np
from collections import defaultdict
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# 1. 載入 20 Newsgroups 數據集
# -------------------------------
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
X = newsgroups.data
y = newsgroups.target

# -------------------------------
# 2. 分割數據集為訓練集和測試集
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 3. 使用 CountVectorizer 將文本轉換成詞頻矩陣，
#    僅保留至少 3 個英文字母的單詞，且擴大詞彙表至 5000
# -------------------------------
vectorizer = CountVectorizer(max_features=5000, stop_words='english', 
                             token_pattern=r'(?u)\b[A-Za-z]{3,}\b')
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# -------------------------------
# 4. 使用 scikit-learn 的 LDA 模型提取主題（純 CPU 運算）
# -------------------------------
n_topics = 10
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X_train_counts)

# -------------------------------
# 5. 印出每個主題前 10 個單字，並建立 concept_words_list
# -------------------------------
feature_names = vectorizer.get_feature_names_out()
n_top_words = 10
concept_words_list = []
for topic_idx, topic in enumerate(lda.components_):
    top_indices = topic.argsort()[::-1][:n_top_words]
    top_words = [feature_names[i] for i in top_indices]
    concept_words_list.append(top_words)
    print(f"Topic #{topic_idx}: {' '.join(top_words)}")

# -------------------------------
# 6. 將每篇文章轉換成主題分佈向量（作為後續分類的特徵）
# -------------------------------
X_train_topics = lda.transform(X_train_counts)
X_test_topics = lda.transform(X_test_counts)

# -------------------------------
# 7. 使用邏輯回歸將主題分佈作為特徵進行分類
# -------------------------------
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_topics, y_train)
y_pred = clf.predict(X_test_topics)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.4f}".format(accuracy))
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))

# -------------------------------
# 8. 定義 Mimno Coherence 計算函數
# -------------------------------
def compute_mimno_coherence(concept_words_list, texts):
    """
    Mimno 和 McCallum (2008) Coherence 計算方式，避免 division by zero 錯誤

    Parameters:
    - concept_words_list: List[List[str]]，每個概念的關鍵詞
    - texts: List[List[str]]，原始文本分詞後的列表

    Returns:
    - coherence_scores: List[float]，每個概念的 Coherence 值
    """
    # 記錄每個詞在哪些文件中出現 (使用 set 儲存文件索引)
    doc_word_counts = defaultdict(set)
    for i, text in enumerate(texts):
        for word in set(text):
            doc_word_counts[word].add(i)

    coherence_scores = []
    for concept_words in concept_words_list:
        pairs = [(w1, w2) for i, w1 in enumerate(concept_words) for w2 in concept_words[i+1:]]
        coherence = []
        for w1, w2 in pairs:
            D_w1 = len(doc_word_counts[w1])  # 包含 w1 的文件數量
            D_w1_w2 = len(doc_word_counts[w1] & doc_word_counts[w2])  # 同時包含 w1 和 w2 的文件數量
            if D_w1 == 0:
                continue  # 若 w1 不出現於任何文件中，跳過
            score = np.log((D_w1_w2 + 1) / (D_w1 + 1e-10))  # 加上小數值以避免除零
            coherence.append(score)
        coherence_scores.append(np.mean(coherence) if coherence else -np.inf)
    return coherence_scores

# -------------------------------
# 9. 準備文本分詞後的列表，用於計算 Coherence
#    這裡利用 vectorizer 的 analyzer 進行分詞
# -------------------------------
analyzer = vectorizer.build_analyzer()
tokenized_texts = [analyzer(doc) for doc in X_train]

# -------------------------------
# 10. 計算並印出 Mimno Coherence 分數
# -------------------------------
mimno_coherence_scores = compute_mimno_coherence(concept_words_list, tokenized_texts)
for i, score in enumerate(mimno_coherence_scores):
    print(f"Concept {i+1} Mimno Coherence: {score:.4f}")