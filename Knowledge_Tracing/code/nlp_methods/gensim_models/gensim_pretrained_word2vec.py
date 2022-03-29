import os.path

import gensim.downloader as downloader
import numpy as np
from time import time
from Knowledge_Tracing.code.models.base_model import base_model
from gensim.models import KeyedVectors


# WORD2VEC using Gensim:

class pretrained_word2vec():
    def __init__(self, load=True, keyedvectors="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/code/models/word2vec_DKT/"
                      "vectors.kv",
                 min_count=2, window=5, vector_size=300, workers=3, sg=1, pretrained='conceptnet-numberbatch-17-06-300'):
        self.name = "pretrained_word2vec_"
        self.min_count = min_count
        self.window = window
        self.workers = workers
        self.sg = sg
        self.epochs = None
        self.pretrained = pretrained
        self.texts_df = None
        self.pro_num = 0
        self.words_num = 0
        if self.pretrained in list(downloader.info()['models'].keys()):
            print("download")
            if load:
                self.wordvectors = KeyedVectors.load(keyedvectors)
            else:
                self.wordvectors = downloader.load(pretrained)
            self.wordvectors.save('vectors.kv')
            self.vector_size = self.wordvectors.vector_size
            print("end")

        self.similarities = None
        self.texts = None

    def load(self, load_path, load_name, download=False):
        if download:
            self.wordvectors = downloader.load(load_path)
            self.name = self.name + load_name
        else:
            self.wordvectors = KeyedVectors.load(load_path)
            self.name = self.name + load_name
        self.wordvectors.save('vectors.kv')

    def transform(self, texts_df, text_column):
        self.texts_df = texts_df
        self.text_column = text_column
        for text, problem in list(zip(list(self.texts_df[self.text_column].values),
                                      list(self.texts_df['problem_id'].values))):
            embedding = np.zeros(shape=self.vector_size)
            flag = False
            for word in text:
                if word in self.wordvectors.vocab:
                    flag = True
                    embedding = embedding + np.array(self.wordvectors.get_vector(word, norm=True))
            if len(text) > 0:
                embedding = embedding / float(len(text))
            norm = np.linalg.norm(embedding)
            if norm > 0.0:
                embedding = embedding / norm
            self.embeddings[problem] = embedding
        self.pro_num = len(self.texts_df['problem_id'])
        self.words_num = self.vector_size

    def save_vectors(self, path="", name="poj"):
        word_vectors = self.wordvectors
        word_vectors.save(path+name+"_word2vec_" + str(self.vector_size) + "_" + str(self.epochs) + ".wordvectors")

    def compute_problem_score(self, input_problems, corrects, target_problem):
        item_scores = self.compute_similarities(input_problems, target_problem)
        item_scores = item_scores.dot(corrects)
        return float(item_scores)

    def get_encoding(self, problem):
        return self.embeddings[problem]
