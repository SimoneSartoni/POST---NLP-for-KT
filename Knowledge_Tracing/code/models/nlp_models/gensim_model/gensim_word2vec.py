import os.path

import numpy as np
from time import time
from gensim.models import Word2Vec


# WORD2VEC using Gensim:

class word2vec:
    def __init__(self, min_count=2, window=5, vector_size=300, workers=3, sg=1):
        self.min_count = min_count
        self.window = window
        self.vector_size = vector_size
        self.workers = workers
        self.sg = sg
        self.epochs = None
        self.name = "gensim_word2vec"
        self.word2vec = Word2Vec(min_count=min_count,
                                 window=window,
                                 vector_size=vector_size,
                                 workers=workers,
                                 sg=sg)
        self.embeddings = {}
        self.word_vectors = None
        self.texts_df = None
        self.text_column = ""
        self.pro_num = 0
        self.words_num = 0

    def load_model(self, epochs, path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files/", name="poj"):
        x = "word2vec_" + str(self.vector_size) + "_" + str(epochs) + ".model"
        path = path + name + '/'
        self.word2vec = Word2Vec.load(path + x)
        self.min_count = self.word2vec.min_count
        self.window = self.word2vec.window
        self.vector_size = self.word2vec.vector_size
        self.workers = self.word2vec.workers
        self.sg = self.word2vec.sg
        self.epochs = self.word2vec.epochs

    def load_word_vectors(self, epochs, path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files/", name="poj"):
        x = "word2vec_" + str(self.vector_size) + "_" + str(epochs) + ".wordvectors"
        path = path + name + '/'
        self.embeddings = Word2Vec.load(path)
        self.min_count = self.word2vec.min_count
        self.window = self.word2vec.window
        self.vector_size = self.word2vec.vector_size
        self.workers = self.word2vec.workers
        self.sg = self.word2vec.sg
        self.epochs = self.word2vec.epochs

    def fit(self, texts_df, text_column='sentence', epochs=20, save_path='', save_name=''):
        t = time()
        self.texts_df = texts_df
        self.epochs = epochs
        self.text_column = text_column
        self.word2vec.build_vocab(list(self.texts_df[self.text_column].values()), progress_per=100)
        self.time_to_build = round((time() - t) / 60, 2)
        print('Time to build vocab: {} mins'.format(self.time_to_build))
        t = time()
        self.word2vec.train(list(self.texts_df[self.text_column].values()), total_examples=self.word2vec.corpus_count,
                            epochs=epochs, report_delay=1)
        self.time_to_train = round((time() - t) / 60, 2)
        print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
        self.word_vectors = self.word2vec.wv
        self.save_vectors(save_path, save_name)
        self.save_model(save_path, save_name)

    def transform(self, texts_df, text_column):
        self.texts_df = texts_df
        self.text_column = text_column
        for problem, text in list(zip(list(self.texts_df[text_column].values()), list(self.texts_df['problem_id'].values()))):
            embedding = np.zeros(shape=self.vector_size)
            for word in text:
                embedding = embedding + np.array(self.word2vec.wv[word])
            if len(text) > 0:
                embedding = embedding / float(len(text))
            norm = np.linalg.norm(embedding)
            embedding = embedding / norm
            self.embeddings[problem] = embedding

        self.pro_num = len(self.texts_df['problem_id'])
        self.words_num = self.vector_size

    def save_model(self, path="", name="poj"):
        self.word2vec.save(path+name+"_word2vec_" + str(self.vector_size) + "_" + str(self.epochs) + ".model")

    def save_vectors(self, path="", name="poj"):
        word_vectors = self.word2vec.wv
        word_vectors.save(path+name+"_word2vec_" + str(self.vector_size) + "_" + str(self.epochs) + ".wordvectors")

    def get_encoding(self, problem, norm=False):
        return self.embeddings[problem]

