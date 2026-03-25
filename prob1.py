import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PyPDF2 import PdfReader
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# TASK 1: DATA PREPARATION
'''
folder_path = "."
documents = []

for file in os.listdir(folder_path):
    if file.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""  # avoid None issues from empty pages
        documents.append(text)  # each PDF becomes one document

print("Total documents:", len(documents))

def clean_text(text):
    text = text.lower()  # normalize everything to lowercase
    text = re.sub(r'([a-z])\.([a-z])', r'\1\2', text)  # fix broken words like "e.g"
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)  # remove numbers, punctuation
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    return text

clean_docs = [clean_text(doc) for doc in documents]

tokens = []
for doc in clean_docs:
    tokens.extend(doc.split())  # flatten all documents into one list of words

print("Total tokens:", len(tokens))
vocab = list(set(tokens))  # unique words only
print("Vocabulary size:", len(vocab))

# Save corpus
with open("corpus.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(tokens))  # useful if you want to reuse data later
'''
with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokens = text.split()

print("Total tokens:", len(tokens))

vocab = set(tokens)
print("Vocabulary size:", len(vocab))


# WordCloud
wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(tokens))
plt.figure(figsize=(10,5))
plt.imshow(wc)
plt.axis("off")
plt.title("Word Cloud")
plt.savefig("wordcloud.png")
plt.show()

# VOCAB MAPPING
word2idx = {w:i for i,w in enumerate(vocab)}  # word → index
idx2word = {i:w for w,i in word2idx.items()}  # index → word
vocab_size = len(vocab)

# NEGATIVE SAMPLING
word_counts = Counter(tokens)  # frequency of each word
total_words = sum(word_counts.values())

# smoothing with power 0.75 (important trick from word2vec paper)
# gives less frequent words a slightly higher chance
word_freq = {w: (c/total_words)**0.75 for w,c in word_counts.items()}
Z = sum(word_freq.values())
word_probs = [word_freq[w]/Z for w in vocab]  # normalized probabilities

def get_negative_samples(k):
    # randomly sample k "noise" words based on frequency distribution
    return torch.tensor(random.choices(range(vocab_size), weights=word_probs, k=k))

# DATA GENERATION
def generate_cbow_data(tokens, window_size):
    data = []
    for i in range(len(tokens)):
        context = []
        for j in range(-window_size, window_size+1):
            if j != 0 and 0 <= i+j < len(tokens):
                context.append(word2idx[tokens[i+j]])
        if context:
            # CBOW: predict target using surrounding words
            data.append((context, word2idx[tokens[i]]))
    return data

def generate_skipgram_data(tokens, window_size):
    data = []
    for i in range(len(tokens)):
        target = word2idx[tokens[i]]
        for j in range(-window_size, window_size+1):
            if j != 0 and 0 <= i+j < len(tokens):
                context = word2idx[tokens[i+j]]
                # SkipGram: predict surrounding words using target
                data.append((target, context))
    return data

# CBOW WITH NEG SAMPLING
class CBOWNeg(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_size)   # input embeddings
        self.out_embed = nn.Embedding(vocab_size, embed_size)  # output embeddings

    def forward(self, context, target, negatives):
        v = torch.mean(self.in_embed(context), dim=0)  # average context vectors → one representation
        u_pos = self.out_embed(target)

        # positive pair should have high similarity
        pos_loss = torch.log(torch.sigmoid(torch.dot(v, u_pos)))

        neg_loss = 0
        for neg in negatives:
            u_neg = self.out_embed(neg)
            # negative pairs should have low similarity
            neg_loss += torch.log(torch.sigmoid(-torch.dot(v, u_neg)))

        return -(pos_loss + neg_loss)  # negative log likelihood (we minimize this)

# SKIPGRAM WITH NEG SAMPLING
class SkipGramNeg(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_size)
        self.out_embed = nn.Embedding(vocab_size, embed_size)

    def forward(self, target, context, negatives):
        v = self.in_embed(target)  # embedding of center word
        u_pos = self.out_embed(context)

        pos_loss = torch.log(torch.sigmoid(torch.dot(v, u_pos)))

        neg_loss = 0
        for neg in negatives:
            u_neg = self.out_embed(neg)
            neg_loss += torch.log(torch.sigmoid(-torch.dot(v, u_neg)))

        return -(pos_loss + neg_loss)

# TASK 2 :TRAINING FUNCTIONS

def train_cbow(data, embed_size=100, epochs=2, neg_samples=5):
    model = CBOWNeg(vocab_size, embed_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        total_loss = 0
        for context, target in data:
            context = torch.tensor(context)
            target = torch.tensor(target)
            negatives = get_negative_samples(neg_samples)

            optimizer.zero_grad()  # clear old gradients
            loss = model(context, target, negatives)
            loss.backward()       # compute gradients
            optimizer.step()      # update weights
            total_loss += loss.item()

        print(f"CBOW Epoch {epoch+1}, Loss: {total_loss:.4f}")  # monitor training

    return model

def train_skipgram(data, embed_size=100, epochs=2, neg_samples=5):
    model = SkipGramNeg(vocab_size, embed_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        total_loss = 0
        for target, context in data:
            target = torch.tensor(target)
            context = torch.tensor(context)
            negatives = get_negative_samples(neg_samples)

            optimizer.zero_grad()
            loss = model(target, context, negatives)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"SkipGram Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return model

# EXPERIMENTS
print("EXPERIMENTS")

embedding_sizes = [50, 100]
window_sizes = [2, 4]
neg_samples_list = [2, 5]

results = []

best_cbow_model = None
best_sg_model = None
best_loss = float("inf")

# trying different combinations to see what works best
for w in window_sizes:
    print(f"\n===== Window Size: {w} =====")

    cbow_data = generate_cbow_data(tokens, w)
    sg_data = generate_skipgram_data(tokens, w)

    for emb in embedding_sizes:
        for neg in neg_samples_list:

            print(f"\nCBOW | emb={emb}, neg={neg}")
            cbow_model = CBOWNeg(vocab_size, emb)
            optimizer = optim.Adam(cbow_model.parameters(), lr=0.01)

            cbow_loss = 0
            for context, target in cbow_data:
                context = torch.tensor(context)
                target = torch.tensor(target)
                negatives = get_negative_samples(neg)

                optimizer.zero_grad()
                loss = cbow_model(context, target, negatives)
                loss.backward()
                optimizer.step()

                cbow_loss += loss.item()

            print("CBOW Loss:", round(cbow_loss,2))

            print(f"SkipGram | emb={emb}, neg={neg}")
            sg_model = SkipGramNeg(vocab_size, emb)
            optimizer = optim.Adam(sg_model.parameters(), lr=0.01)

            sg_loss = 0
            for target, context in sg_data:
                target = torch.tensor(target)
                context = torch.tensor(context)
                negatives = get_negative_samples(neg)

                optimizer.zero_grad()
                loss = sg_model(target, context, negatives)
                loss.backward()
                optimizer.step()

                sg_loss += loss.item()

            print("SkipGram Loss:", round(sg_loss,2))

            results.append({
                "window": w,
                "embedding": emb,
                "neg": neg,
                "cbow_loss": round(cbow_loss,2),
                "sg_loss": round(sg_loss,2)
            })

            # Save best model (based on SkipGram loss)
            if sg_loss < best_loss:
                best_loss = sg_loss
                best_cbow_model = cbow_model
                best_sg_model = sg_model

# Use best models for further tasks
cbow_model = best_cbow_model
sg_model = best_sg_model

# TASK 3:SIMILARITY + ANALOGY
def cosine_sim(a, b):
    return F.cosine_similarity(a, b, dim=0).item()

def get_embed(model, word):
    return model.in_embed.weight[word2idx[word]]  # fetch embedding vector

def nearest(word, model):
    if word not in word2idx:
        return "Not found"

    vec = get_embed(model, word)
    sims = []

    for w in vocab:
        if w == word:
            continue
        sim = cosine_sim(vec, get_embed(model, w))
        sims.append((w, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:5]  # top 5 most similar words

words = ["research", "student", "phd", "exam"]

print("\nCBOW Neighbors:")
for w in words:
    print(w, nearest(w, cbow_model))

print("\nSkipGram Neighbors:")
for w in words:
    print(w, nearest(w, sg_model))

def analogy(a, b, c, model):
    # idea: a is to b as c is to ?
    vec = get_embed(model, b) - get_embed(model, a) + get_embed(model, c)
    sims = []

    for w in vocab:
        if w in [a,b,c]:
            continue
        sims.append((w, cosine_sim(vec, get_embed(model, w))))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:3]

print("\nAnalogy (CBOW):", analogy("ug", "btech", "pg", cbow_model))
print("\nAnalogy (SkipGram):", analogy("ug", "btech", "pg", sg_model))

# T-SNE VISUALIZATION
def plot_tsne(words, model, title):
    vectors = []
    valid = []

    for w in words:
        if w in word2idx:
            vectors.append(get_embed(model, w).detach().numpy())
            valid.append(w)

    # reduce high-dim embeddings to 2D for visualization
    tsne = TSNE(n_components=2, perplexity=2, random_state=42)
    vectors = np.array(vectors)
    coords = tsne.fit_transform(vectors)

    plt.figure()
    for i, word in enumerate(valid):
        plt.scatter(coords[i,0], coords[i,1])
        plt.text(coords[i,0], coords[i,1], word)

    plt.title(title)
    plt.savefig(title + ".png")
    plt.show()

plot_tsne(words, cbow_model, "CBOW_TSNE")
plot_tsne(words, sg_model, "SkipGram_TSNE")

# GENSiM IMPLEMENTATION

from gensim.models import Word2Vec

sentences = [tokens[i:i+50] for i in range(0, len(tokens), 50)]

# built-in optimized implementation (faster + industry standard)
cbow_gensim = Word2Vec(sentences, vector_size=100, window=4, sg=0, negative=5)
sg_gensim = Word2Vec(sentences, vector_size=100, window=4, sg=1, negative=5)

print("\nGensim CBOW:", cbow_gensim.wv.most_similar("research"))
print("Gensim SkipGram:", sg_gensim.wv.most_similar("research"))

print("\nGensim Analogy:",
      cbow_gensim.wv.most_similar(positive=["pg","btech"], negative=["ug"]))
