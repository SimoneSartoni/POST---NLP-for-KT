import os.path

import numpy as np
from time import time
from gensim.models import Word2Vec
from Knowledge_Tracing.code.models.base_model import base_model


# WORD2VEC using Gensim:

class word2vec(base_model):
    def __init__(self, min_count=2, window=5, vector_size=300, workers=3, sg=1):
        super(word2vec, self).__init__("gensim_word2vec", "NLP")
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
        self.similarities = None
        self.texts_df = None
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
        self.wordvectors = Word2Vec.load(path)
        self.min_count = self.word2vec.min_count
        self.window = self.word2vec.window
        self.vector_size = self.word2vec.vector_size
        self.workers = self.word2vec.workers
        self.sg = self.word2vec.sg
        self.epochs = self.word2vec.epochs

    def get_similarity_matrix_from_vectors(self, word_vectors):
        self.similarities = np.dot(word_vectors.vectors, word_vectors.vectors.T)

    def fit(self, epochs=20, path='', name=''):
        t = time()
        self.epochs = epochs
        self.word2vec.build_vocab(self.texts_df, progress_per=100)
        self.time_to_build = round((time() - t) / 60, 2)
        print('Time to build vocab: {} mins'.format(self.time_to_build))
        t = time()
        self.word2vec.train(self.texts_df, total_examples=self.word2vec.corpus_count, epochs=epochs, report_delay=1)
        self.time_to_train = round((time() - t) / 60, 2)
        print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
        self.wordvectors = self.word2vec.wv
        self.save_vectors(path, name)
        self.save_model(path, name)

    def encode_problems(self, texts_df):
        self.texts_df = texts_df
        self.pro_num = len(self.texts_df['problem_id'])
        self.words_num = self.vector_size

    def save_model(self, path="", name="poj"):
        self.word2vec.save(path+name+"_word2vec_" + str(self.vector_size) + "_" + str(self.epochs) + ".model")

    def save_vectors(self, path="", name="poj"):
        word_vectors = self.word2vec.wv
        word_vectors.save(path+name+"_word2vec_" + str(self.vector_size) + "_" + str(self.epochs) + ".wordvectors")

    def compute_problem_score(self, input_problems, corrects, target_problem):
        item_scores = self.compute_similarities(input_problems, target_problem)
        item_scores = item_scores.dot(corrects)
        return float(item_scores)

    def compute_similarities(self, input_problems, target_problem):
        """
        """
        item_scores = []
        final_score = 0.
        if target_problem in self.problem_id_to_index.keys():
            for input_id in input_problems:
                if input_id in self.problem_id_to_index.keys() and len(self.problem_to_text[input_id]) > 0 and len(
                        self.problem_to_text[target_problem]) > 0:
                    similarity = self.wordvectors.n_similarity(self.problem_to_text[input_id],
                                                               self.problem_to_text[target_problem])
                else:
                    similarity = 0.0
                item_scores.append(similarity)
            final_score = np.array(item_scores).transpose()
        return final_score

    def compute_encoding(self, input_problems, corrects, target_problem):
        pos_mean_encoding = np.zeros(shape=self.vector_size, dtype=np.float)
        neg_mean_encoding = np.zeros(shape=self.vector_size, dtype=np.float)
        pos, neg = 0.0, 0.0
        for p, c in list(zip(input_problems, corrects)):
            if p in self.problem_id_to_index.keys():
                # and p not in unique_problems_set:
                # unique_problems_set.add(p)
                text = self.problem_to_text[p]
                sentence_encoding = np.zeros(shape=self.vector_size)
                for word in text:
                    sentence_encoding = sentence_encoding + np.array(self.word2vec.wv[word])
                if len(text) > 0:
                    sentence_encoding = sentence_encoding / float(len(text))
                if c > 0.0:
                    pos_mean_encoding = pos_mean_encoding + sentence_encoding
                    pos += 1.0
                else:
                    neg += 1.0
                    neg_mean_encoding = neg_mean_encoding + sentence_encoding
        if pos > 0.0:
            pos_mean_encoding = pos_mean_encoding / float(pos)
        if neg > 0.0:
            neg_mean_encoding = neg_mean_encoding / float(neg)
        target_encoding = np.zeros(shape=self.vector_size, dtype=np.float)
        if target_problem in self.problem_id_to_index.keys():
            text = self.problem_to_text[target_problem]
            target_encoding = np.zeros(shape=self.vector_size)
            for word in text:
                target_encoding = target_encoding + np.array(self.word2vec.wv[word])
            if len(text) > 0:
                target_encoding = target_encoding / float(len(text))
        encoding = np.concatenate((pos_mean_encoding, neg_mean_encoding, target_encoding), axis=0)
        return encoding

    def get_encoding(self, problem, norm=False):
        row = self.texts_df.loc[self.texts_df['question_id'] == problem]
        sentence_encoding = np.zeros(shape=self.vector_size)
        num = 0
        for word in row['body'].values[0]:
            if word in self.wordvectors.key_to_index:
                sentence_encoding = sentence_encoding + np.array(self.wordvectors.get_vector(word, norm=norm))
                num += 1
        if num > 0:
            sentence_encoding = sentence_encoding / float(num)
        return sentence_encoding

    def get_serializable_params(self):
        return {"name": self.name, "min_count": self.min_count, "window": self.window, "vector_size": self.vector_size,
                "sg": self.sg, "epochs": self.epochs}
