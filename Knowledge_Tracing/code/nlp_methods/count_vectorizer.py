import numpy as np
import os
from scipy import sparse as sps

from sklearn.feature_extraction.text import CountVectorizer


def write_txt(file, data):
    with open(file, 'w') as f:
        for dd in data:
            f.write(str(dd) + '\n')


def identity_tokenizer(text):
    return text


class count_vectorizer:
    def __init__(self, max_df, min_df, binary, max_features=None):
        self.count_vectorizer = CountVectorizer(
            analyzer='word',
            tokenizer=identity_tokenizer,
            preprocessor=identity_tokenizer,
            token_pattern=None,
            max_df=max_df,
            min_df=min_df,
            binary=binary,
            max_features=max_features
        )
        self.min_df = min_df
        self.max_df = max_df
        self.binary = binary
        self.max_features = max_features
        self.pro_num = None
        self.words_num = None
        self.embeddings = {}
        self.texts_df = None
        self.vector_size = 0
        self.name = "count_vectorizer"

    def fit(self, texts_df, save_filepath='./'):
        self.texts_df = texts_df
        self.count_vectorizer = self.count_vectorizer.fit(self.texts_df['list_of_words'])

        # Save sparse matrix in current directory
        embeddings = self.count_vectorizer.transform(self.texts_df['list_of_words'])
        for key, embedding in list(zip(list(self.texts_df['problem_id'].values), embeddings)):
            self.embeddings[key] = embedding
        self.vector_size = embeddings.shape[1]
        self.pro_num = len(self.texts_df['problem_id'])
        self.words_num = self.vector_size
        print(list(self.embeddings.keys()))

    def get_encoding(self, problem):
        encoding = np.array(self.embeddings[problem].todense()).squeeze()
        return encoding
