import os.path

import numpy as np
from time import time
from gensim.models import Word2Vec
from Knowledge_Tracing.code.models.base_model import base_model


# WORD2VEC using Gensim:

class world2vec(base_model):
    def __init__(self, name, class_of_method, min_count=2, window=5, vector_size=300, workers=3, sg=1):
        super(world2vec, self).__init__(name, class_of_method)
        self.min_count = min_count
        self.window = window
        self.vector_size = vector_size
        self.workers = workers
        self.sg = sg
        self.epochs = None
        self.word2vec = Word2Vec(min_count=min_count,
                                 window=window,
                                 vector_size=vector_size,
                                 workers=workers,
                                 sg=sg)
        self.wordvectors = None
        self.problem_to_text = {}
        self.problem_id_to_index = None
        self.problem_ids = None

    def load_model(self, epochs, path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/", name="poj"):
        x = "word2vec_" + str(self.vector_size) + "_" + str(epochs) + ".model"
        path = path + name + '/'
        self.word2vec = Word2Vec.load(path + x)
        self.min_count = self.word2vec.min_count
        self.window = self.word2vec.window
        self.vector_size = self.word2vec.vector_size
        self.workers = self.word2vec.workers
        self.sg = self.word2vec.sg
        self.epochs = self.word2vec.epochs

    def load_word_vectors(self, epochs, path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/", name="poj"):
        x = "word2vec_" + str(self.vector_size) + "_" + str(epochs) + ".wordvectors"
        path = path + name + '/'
        self.wordvectors = Word2Vec.load(path)
        self.min_count = self.word2vec.min_count
        self.window = self.word2vec.window
        self.vector_size = self.word2vec.vector_size
        self.workers = self.word2vec.workers
        self.sg = self.word2vec.sg
        self.epochs = self.word2vec.epochs

    def get_similarity_matrix_from_vectors(self, word_vectors):
        self.word2vec = np.dot(word_vectors.vectors, word_vectors.vectors.T)

    def fit(self, texts, epochs=10, path='', name=''):
        t = time()
        self.epochs = epochs
        self.word2vec.build_vocab(texts, progress_per=100)
        self.time_to_build = round((time() - t) / 60, 2)
        print('Time to build vocab: {} mins'.format(self.time_to_build))
        t = time()
        self.word2vec.train(texts, total_examples=self.word2vec.corpus_count, epochs=epochs, report_delay=1)
        self.time_to_train = round((time() - t) / 60, 2)
        print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
        self.wordvectors = self.word2vec.wv
        self.save_vectors(path, name)
        self.save_model(path, name)

    def encode_problems(self, problem_id_to_index, texts):
        self.problem_id_to_index = problem_id_to_index
        self.problem_ids = problem_id_to_index.keys()
        vocabulary = self.wordvectors.key_to_index
        k = 0
        for problem_id in self.problem_ids:
            problem_words = []
            t = texts[self.problem_id_to_index[problem_id]]
            for word in t:
                if word in vocabulary:
                    problem_words.append(word)
            self.problem_to_text[problem_id] = problem_words
            k += 1

    def save_model(self, path, name):
        self.word2vec.save("word2vec_" + str(self.vector_size) + "_" + str(self.epochs) + ".model")

    def save_vectors(self, name, path):
        word_vectors = self.word2vec.wv
        word_vectors.save("word2vec_" + str(self.vector_size) + "_" + str(self.epochs) + ".wordvectors")

    # Overriding abstract method
    def compute_problem_score(self, input_problems, corrects, target_problem, default_value):
        """

        """
        item_scores = []
        input_ids = []
        correct_ids = []
        for p, c in list(zip(input_problems, corrects)):
            if p in self.problem_to_text:
                input_ids.append(p)
                correct_ids.append(c)
        if target_problem in self.problem_to_text:
            for input_id in input_ids:
                if input_id in self.problem_to_text and len(self.problem_to_text[input_id]) > 0 and len(
                        self.problem_to_text[target_problem]) > 0:
                    similarity = self.wordvectors.n_similarity(self.problem_to_text[input_id],
                                                               self.problem_to_text[target_problem])
                else:
                    similarity = 0.0
                item_scores.append(similarity)
            item_scores = np.array(item_scores).transpose().dot(correct_ids)
            if item_scores and item_scores > 0:
                return 1.0
            elif item_scores and item_scores < 0:
                return 0.0
        return default_value
