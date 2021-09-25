import os.path

import numpy as np
from time import time
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from Knowledge_Tracing.code.models.base_model import base_model


# DOC2VEC using Gensim:

class doc2vec(base_model):
    def __init__(self, min_count=2, window=5, vector_size=300, workers=3, sg=1):
        super(doc2vec, self).__init__("gensim_doc2vec", "NLP")
        self.min_count = min_count
        self.window = window
        self.vector_size = vector_size
        self.workers = workers
        self.sg = sg
        self.epochs = None
        self.doc2vec = Doc2Vec(min_count=min_count,
                               window=window,
                               vector_size=vector_size,
                               workers=workers,
                               sg=sg)
        self.docvectors = None
        self.problem_to_text = {}
        self.problem_id_to_index = None

    def load_model(self, epochs,
                   path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files/",
                   name="poj"):
        x = "word2vec_" + str(self.vector_size) + "_" + str(epochs) + ".model"
        path = path + name + '/'
        self.doc2vec = Doc2Vec.load(path + x)
        self.min_count = self.doc2vec.min_count
        self.window = self.doc2vec.window
        self.vector_size = self.doc2vec.vector_size
        self.workers = self.doc2vec.workers
        self.sg = self.doc2vec.sg
        self.epochs = self.doc2vec.epochs

    def load_word_vectors(self, epochs,
                          path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files/",
                          name="poj"):
        x = "word2vec_" + str(self.vector_size) + "_" + str(epochs) + ".wordvectors"
        path = path + name + '/'
        self.docvectors = Doc2Vec.load(path)
        self.min_count = self.doc2vec.min_count
        self.window = self.doc2vec.window
        self.vector_size = self.doc2vec.vector_size
        self.workers = self.doc2vec.workers
        self.sg = self.doc2vec.sg
        self.epochs = self.doc2vec.epochs

    def get_similarity_matrix_from_vectors(self, word_vectors):
        self.doc2vec = np.dot(word_vectors.vectors, word_vectors.vectors.T)

    def fit(self, texts, problem_id_to_index, epochs=20, path='', name=''):
        t = time()
        self.epochs = epochs
        taggedDocuments = []
        for index in problem_id_to_index.keys():
            taggedDocuments.append(TaggedDocument(index, texts[problem_id_to_index[index]]))

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

    def save_model(self, path="", name="poj"):
        self.doc2vec.save(path + name + "_word2vec_" + str(self.vector_size) + "_" + str(self.epochs) + ".model")

    def save_vectors(self, path="", name="poj"):
        word_vectors = self.doc2vec.wv
        word_vectors.save(path + name + "_word2vec_" + str(self.vector_size) + "_" + str(self.epochs) + ".wordvectors")

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
                if input_id in self.problem_id_to_index.keys():
                    similarity = self.doc2vec.similarity_unseen_docs(self.docvectors[input_id],
                                                                     self.docvectors[target_problem])
                else:
                    similarity = 0.0
                item_scores.append(similarity)
            final_score = np.array(item_scores).transpose()
        return final_score

    def compute_encoding(self, input_problems, corrects, target_problem):
        pos_mean_encoding = np.zeros(shape=self.words_num, dtype=np.float)
        neg_mean_encoding = np.zeros(shape=self.words_num, dtype=np.float)
        pos, neg = 0.0, 0.0
        for p, c in list(zip(input_problems, corrects)):
            if p in self.problem_id_to_index.keys():
                # and p not in unique_problems_set:
                # unique_problems_set.add(p)
                if c > 0.0:
                    pos += 1.0
                    x = np.array(self.docvectors[p])
                    pos_mean_encoding = pos_mean_encoding + x
                else:
                    neg += 1.0
                    neg_mean_encoding = neg_mean_encoding + np.array(self.docvectors[p])
        if pos > 0.0:
            pos_mean_encoding = pos_mean_encoding / pos
        if neg > 0.0:
            neg_mean_encoding = neg_mean_encoding / neg
        target_encoding = np.zeros(shape=self.words_num, dtype=np.float)
        if target_problem in self.problem_id_to_index.keys():
            x = np.array(self.docvectors[target_problem])
            target_encoding = target_encoding + x
        encoding = np.concatenate((pos_mean_encoding, neg_mean_encoding, target_encoding), axis=0)
        return encoding

    def get_serializable_params(self):
        return {"name": self.name, "min_count": self.min_count, "window": self.window, "vector_size": self.vector_size,
                "sg": self.sg, "epochs": self.epochs}
