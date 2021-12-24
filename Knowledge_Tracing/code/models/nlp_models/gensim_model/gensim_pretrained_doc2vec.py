import os.path

import gensim.downloader as downloader
import numpy as np
from time import time
from Knowledge_Tracing.code.models.base_model import base_model
from Knowledge_Tracing.code.Notebook.gensim_develop import gensim_pretrained as gensim_dir
from Knowledge_Tracing.code.Notebook.gensim_develop.gensim_pretrained.models.doc2vec import TaggedDocument
pretrained_emb = "word2vec_pretrained.txt"  #This is a pretrained word2vec model of C text format

# WORD2VEC using Gensim:


class pretrained_doc2vec(base_model):

    def __init__(self, min_count=2, window=5, vector_size=300, workers=3, sg=1, pretrained=pretrained_emb):
        super(pretrained_doc2vec, self).__init__("pretrained_doc2vec"+pretrained, "NLP")
        self.min_count = min_count
        self.window = window
        self.workers = workers
        self.sg = sg
        self.epochs = None
        self.pretrained = pretrained
        print("download")
        self.doc2vec = gensim_dir.models.doc2vec.Doc2Vec(vector_size=300, min_count=1, epochs=20, dm=0,
                                                         pretrained_emb=pretrained_emb)
        self.vector_size = self.doc2vec.vector_size
        print("end")
        self.problem_to_text = {}
        self.problem_id_to_index = None
        self.similarities = None
        self.texts = None

    def get_similarity_matrix_from_vectors(self, word_vectors):
        self.similarities = np.dot(word_vectors.vectors, word_vectors.vectors.T)

    def fit(self, texts, problem_id_to_index, epochs=20, path='', name=''):
        t = time()
        self.epochs = epochs
        taggedDocuments = []
        self.problem_id_to_index = problem_id_to_index
        for index in problem_id_to_index.keys():
            taggedDocuments.append(TaggedDocument(texts[problem_id_to_index[index]], [index]))

        self.doc2vec.build_vocab(taggedDocuments, progress_per=100)
        self.time_to_build = round((time() - t) / 60, 2)
        print('Time to build vocab: {} mins'.format(self.time_to_build))
        t = time()
        self.doc2vec.train(taggedDocuments, total_examples=self.doc2vec.corpus_count, epochs=epochs, report_delay=1)
        self.time_to_train = round((time() - t) / 60, 2)
        print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
        self.docvectors = self.doc2vec.dv
        # self.save_vectors(path, name)
        # self.save_model(path, name)

    def encode_problems(self, problem_id_to_index, texts):
        self.problem_id_to_index = problem_id_to_index
        k = 0
        self.texts = {}
        vocabulary = set(self.wordvectors.index_to_key)
        print(vocabulary)
        for p in problem_id_to_index.keys():
            if k == 0:
                print(texts[self.problem_id_to_index[p]])
                print(set(texts[self.problem_id_to_index[p]]))
                print(set(texts[self.problem_id_to_index[p]]).intersection(vocabulary))
            self.texts[p] = list(set(texts[self.problem_id_to_index[p]]).intersection(vocabulary))

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

    def get_serializable_params(self):
        return {"name": self.name, "min_count": self.min_count, "window": self.window, "vector_size": self.vector_size,
                "sg": self.sg, "epochs": self.epochs}
