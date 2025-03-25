import numpy as np
import torch
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from collections import Counter
from gdcm import GuidedDiverseConceptMiner

#nltk.download("punkt")
#nltk.download("stopwords")

def preprocess_pipeline(raw_texts, labels, max_vocab_size=20000, window_size=5, pad_token_idx=0, label_encoder=None
                        , vocab=None, id2word=None):
    #full preprocessing pipeline：tokenize → build vocab → build BoW → build window

    assert len(raw_texts) == len(labels)
    assert window_size % 2 == 1

    #Tokenization 分詞
    valid_data = [
        (i, word_tokenize(doc.lower())) for i, doc in enumerate(raw_texts)
        if isinstance(doc, str) and doc.strip()
    ]
    tokenized_texts = [tokens for _, tokens in valid_data]
    valid_indices = [i for i, _ in valid_data]
    labels = [labels[i] for i in valid_indices]

    print("Tokenization done")

    #StopWord and Not Alpha
    stop_words = set(stopwords.words("english"))
    cleaned_texts = [
        [word for word in doc if word.isalpha() and word not in stop_words]
        for doc in tokenized_texts
    ]

    print("Stopwrod done")

    #Build Vocab
    if vocab is None:
        word_counts = Counter(word for doc in cleaned_texts for word in doc)
        most_common = word_counts.most_common(max_vocab_size)
        vocab = {word: idx for idx, (word, _) in enumerate(most_common)}
        id2word = {idx: word for word, idx in vocab.items()}
        print("Vocab built from scratch")
    else:
        word_counts = Counter()  # 空的，也OK
        print("Using existing vocab")

    print("Vocab done")

    #Label decoding
    if label_encoder is not None:
        labels_str = label_encoder.inverse_transform(labels)
    else:
        labels_str = labels

    #Build Bag of Words
    X_Bow = np.zeros((len(cleaned_texts), len(vocab)), dtype=np.int64)
    for i, doc in enumerate(cleaned_texts):
        for word in doc:
            if word in vocab:
                X_Bow[i, vocab[word]] += 1
    print("bag-of-words done")
            
    #Build Window
    half_window = window_size // 2
    context_size = 2 * half_window
    doc_windows = []

    for doc_idx, tokens in enumerate(cleaned_texts):
        indexed_tokens = [vocab[word] for word in tokens if word in vocab]
        if len(indexed_tokens) >= 2:
            for i in range(len(indexed_tokens)):
                pivot = indexed_tokens[i]
                start = max(0, i - half_window)
                end = min(len(indexed_tokens), i + half_window + 1)
                context = indexed_tokens[start:i] + indexed_tokens[i+1:end]
                while len(context) < context_size:
                    context.append(pad_token_idx)
                doc_windows.append([doc_idx, pivot] + context + [labels[doc_idx]])
    print("window done")
    
    return tokenized_texts, vocab, id2word, word_counts, X_Bow, np.array(doc_windows, dtype=np.int64), np.array(labels), np.array(labels_str)

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # 確保輸出目錄存在
    output_dir = "newsgroup_test"
    os.makedirs(output_dir, exist_ok=True)

    # 下載 20 Newsgroups 資料集
    categories = None
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))
    raw_texts_train = newsgroups_train.data
    raw_texts_test = newsgroups_test.data

    # 將數值 label 轉回文字 label
    label_names = newsgroups_train.target_names  # e.g., ['alt.atheism', 'comp.graphics', ..., 'talk.religion.misc']
    labels_str_train = [label_names[i] for i in newsgroups_train.target]
    labels_str_test = [label_names[i] for i in newsgroups_test.target]

    # 現在才用 LabelEncoder 編碼（保證對應到文字）
    le = LabelEncoder()
    labels_encoded_train = le.fit_transform(labels_str_train)
    labels_encoded_test = le.transform(labels_str_test)  # 注意：測試集只能 transform，不能 fit！
    
    tokenized_texts_train, vocab, id2word, word_counts, X_train, doc_windows_train, y_train_int, y_train_str = preprocess_pipeline(
        raw_texts=newsgroups_train.data,
        labels=labels_encoded_train,
        max_vocab_size=20000,
        window_size=5,
        label_encoder=le
    )

    tokenized_texts_test, _, _, _, X_test, doc_windows_test, y_test_int, y_test_str = preprocess_pipeline(
        raw_texts=newsgroups_test.data,
        labels=labels_encoded_test,
        window_size=5,
        label_encoder=le,
        vocab=vocab,
        id2word=id2word
    )

    '''device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train, dtype=torch.long, device=device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.long, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

    print(X_train_tensor.shape)
    print(X_test_tensor.shape)
    print(y_train_tensor.shape)
    print(y_test_tensor.shape)'''

    gdcm = GuidedDiverseConceptMiner(
        out_dir=output_dir, embed_dim=300, nnegs=15, nconcepts=25, lam=100.0, rho=100.0, eta=1.0,
        doc_concept_probs=None, word_vectors=None, theta=None, gpu=None,
        inductive=True, inductive_dropout=0.01, hidden_size=100, num_layers=1,
        bow_train=X_train, y_train=y_train_str, bow_test=X_test, y_test=y_test_str, 
        doc_windows=doc_windows_train, vocab=vocab,
        word_counts=word_counts, doc_lens=None, expvars_train=None, expvars_test=None, file_log=False, norm=None
    )

    gdcm.y_train = y_train_int
    gdcm.y_test = y_test_int

    gdcm.fit()