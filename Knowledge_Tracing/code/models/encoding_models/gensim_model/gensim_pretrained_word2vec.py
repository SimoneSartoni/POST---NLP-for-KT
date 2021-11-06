import os.path

import gensim.downloader as downloader
import numpy as np
from time import time
from Knowledge_Tracing.code.models.base_model import base_model
from gensim.models import KeyedVectors


# WORD2VEC using Gensim:

class pretrained_word2vec(base_model):
    def __init__(self, load=True, keyedvectors="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/code/models/word2vec_DKT/"
                      "vectors.kv",
                 min_count=2, window=5, vector_size=300, workers=3, sg=1, pretrained='conceptnet-numberbatch-17-06-300'):
        super(pretrained_word2vec, self).__init__("pretrained_world2vec_"+pretrained, "NLP")
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

    def get_similarity_matrix_from_vectors(self, word_vectors):
        self.similarities = np.dot(word_vectors.vectors, word_vectors.vectors.T)

    def fit(self, epochs=20, path='', name=''):
        t = time()
        self.epochs = epochs
        self.time_to_build = round((time() - t) / 60, 2)
        print('Time to build vocab: {} mins'.format(self.time_to_build))
        t = time()
        self.time_to_train = round((time() - t) / 60, 2)
        print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
        # self.save_vectors(path, name)
        # self.save_model(path, name)

    def encode_problems(self, texts_df):
        self.texts_df = texts_df

        self.pro_num = len(self.texts_df['problem_id'])
        self.words_num = self.vector_size

    def save_vectors(self, path="", name="poj"):
        word_vectors = self.wordvectors
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
                if input_id in self.problem_id_to_index.keys() and len(self.texts[input_id]) > 0 and \
                        len(self.texts[target_problem]) > 0:
                    similarity = self.wordvectors.n_similarity(self.texts[input_id],
                                                               self.texts[target_problem])
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
                text = self.texts[p]
                sentence_encoding = np.zeros(shape=self.vector_size)
                num = 0
                for word in text:
                    sentence_encoding = sentence_encoding + np.array(self.wordvectors[word])
                    num += 1
                if len(text) > 0:
                    sentence_encoding = sentence_encoding / float(num)
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
            text = self.texts[target_problem]
            target_encoding = np.zeros(shape=self.vector_size)
            num = 0
            for word in text:
                target_encoding = target_encoding + np.array(self.wordvectors[word])
                num += 1
            if num > 0:
                target_encoding = target_encoding / float(num)
        encoding = np.concatenate((pos_mean_encoding, neg_mean_encoding, target_encoding), axis=0)
        return encoding

    def get_encoding(self, problem):
        row = self.texts_df.loc[self.texts_df['problem_id']==problem]
        sentence_encoding = np.zeros(shape=self.vector_size)
        num = 0
        for word in row['body'].values:
            sentence_encoding = sentence_encoding + np.array(self.wordvectors[word])
            num += 1
        if len(row['body']) > 0:
            sentence_encoding = sentence_encoding / float(num)
        return sentence_encoding

    def get_serializable_params(self):
        return {"name": self.name, "min_count": self.min_count, "window": self.window, "vector_size": self.vector_size,
                "sg": self.sg, "epochs": self.epochs}
