import os.path

import numpy as np
from time import time
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from Knowledge_Tracing.code.models.base_model import base_model


# DOC2VEC using Gensim:

class doc2vec:
    def __init__(self, min_count=2, window=5, vector_size=300, workers=3, dm=1):
        self.name = "doc2vec"
        self.min_count = min_count
        self.window = window
        self.vector_size = vector_size
        self.workers = workers
        self.epochs = None
        self.doc2vec = Doc2Vec(dm=dm, alpha=0.1, min_alpha=0.025, vector_size=vector_size, workers=workers)
        self.embeddings = {}
        self.texts_df = None
        self.text_column = ""
        self.pro_num = 0
        self.words_num = 0

    """def load_model(self, epochs,
                   path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files/",
                   name="poj"):
        x = "word2vec_" + str(self.vector_size) + "_" + str(epochs) + ".model"
        path = path + name + '/'
        self.doc2vec = Doc2Vec.load(path + x)
        self.min_count = self.doc2vec.min_count
        self.window = self.doc2vec.window
        self.vector_size = self.doc2vec.vector_size
        self.workers = self.doc2vec.workers
        self.epochs = self.doc2vec.epochs

    def load_word_vectors(self, epochs,
                          path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files/",
                          name="poj"):
        x = "word2vec_" + str(self.vector_size) + "_" + str(epochs) + ".wordvectors"
        path = path + name + '/'
        docvectors = Doc2Vec.load(path)
        self.min_count = self.doc2vec.min_count
        self.window = self.doc2vec.window
        self.vector_size = self.doc2vec.vector_size
        self.workers = self.doc2vec.workers
        self.epochs = self.doc2vec.epochs
        self.time_to_build = 0.0"""

    def fit(self, texts_df, text_column="plain_text", epochs=20, save_path='', save_name=''):
        t = time()
        self.epochs = epochs
        self.text_column = text_column
        self.texts_df = texts_df
        tagged_documents = []
        for text, problem in list(zip(list(self.texts_df[text_column].values), list(self.texts_df['problem_id'].values))):
            tagged_documents.append(TaggedDocument(text, problem))
        self.doc2vec.build_vocab(tagged_documents, progress_per=100)
        self.time_to_build = round((time() - t) / 60, 2)
        print('Time to build vocab: {} mins'.format(self.time_to_build))
        t = time()
        self.doc2vec.train(tagged_documents, total_examples=self.doc2vec.corpus_count, epochs=epochs, report_delay=1)
        self.time_to_train = round((time() - t) / 60, 2)
        print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
        self.save_vectors(save_path, save_name)
        self.save_model(save_path, save_name)

    def save_model(self, path="", name="poj"):
        self.doc2vec.save(path + name + "_word2vec_" + str(self.vector_size) + "_" + str(self.epochs) + ".model")

    def save_vectors(self, path="", name="poj"):
        word_vectors = self.doc2vec.wv
        word_vectors.save(path + name + "_word2vec_" + str(self.vector_size) + "_" + str(self.epochs) + ".wordvectors")

    def transform(self, texts_df, text_column):
        self.texts_df = texts_df
        self.text_column = text_column
        for text, problem in list(zip(list(self.texts_df[text_column].values), list(self.texts_df['problem_id'].values))):
            self.embeddings[problem] = np.array(self.doc2vec.dw[problem])

        self.pro_num = len(self.texts_df['problem_id'])
        self.words_num = self.vector_size

    def get_encoding(self, problem):
        return self.embeddings[problem]
