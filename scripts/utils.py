from collections import defaultdict
import pandas as pd
import numpy as np


class Vocab(object):
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.unknown = '<unk>'
        self.add_word(self.unknown, count=0)

    def add_word(self, word, count=1):
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
        self.word_freq[word] += count

    def construct(self, words):
        for word in words:
            self.add_word(word)
        self.total_words = float(sum(self.word_freq.values()))
        print '{} total words with {} uniques'.format(self.total_words, len(self.word_freq))

    def encode(self, word):
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index[word]

    def decode(self, index):
        return self.index_to_word[index]

    def __len__(self):
        return len(self.word_freq)

class Vocab_pre_trained(object):
    def __init__(self, embed_file, words):
        df = pd.read_csv(embed_file, header=None, sep='\s+', engine='python', index_col=0)
        unk_word = df.loc['unk']
        df = df.loc[words].dropna()
        df = df.append(unk_word)
        self.indeces = df.index
        self.pre_trained_embeddings = np.array(df)
        print self.pre_trained_embeddings.shape

    def encode(self, word):
        try:
            index = self.indeces.get_loc(word)
        except:
            index = self.indeces.get_loc('unk')
        return index


import pickle
    
class Vocab_pre_trained_big(object):
    def __init__(self, embed_file, w2idx_file, arabic=False):
        self.arabic = arabic
        self.pre_trained_embeddings = np.load(embed_file)
        with open(w2idx_file, 'rb') as f:
            self.w2idx = pickle.load(f)

    def encode(self, word):
        if self.arabic:
            word = word.decode('utf8')
        try:
            index = self.w2idx[word]
        except:
            index = self.w2idx['<unk>']
        return index