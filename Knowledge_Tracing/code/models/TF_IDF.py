import pandas as pd
import numpy as np
import os
import scipy as sps
from Knowledge_Tracing.code.Similarity.Compute_Similarity import Compute_Similarity

from sklearn.feature_extraction.text import TfidfVectorizer
import base_model


def write_txt(file, data):
    with open(file, 'w') as f:
        for dd in data:
            f.write(str(dd) + '\n')

def identity_tokenizer(text):
    return text


class TF_IDF(base_model):
    def __init__(self, name, type):
        super().__init__(name, type)
        self.tf_idf_vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=identity_tokenizer,
            preprocessor=identity_tokenizer,
            token_pattern=None,
            use_idf=True)
        self.problem_to_text = None
        self.similarity_matrix = None
        self.words_unique = None
        self.pro_num = None
        self.words_num = None
        self.words_dict = None
        self.topK = 100
        self.shrink = 10
        self.normalize = True
        self.similarity = "cosine"



    def fit(self, texts):
        tfidf_vectorizer_vectors = self.tfidf_vectorizer.fit_transform(texts)
        df_tf_idf = pd.DataFrame.sparse.from_spmatrix(tfidf_vectorizer_vectors)
        dataframe_tf_idf = df_tf_idf
        self.words_unique = self.tfidf_vectorizer.get_feature_names()
        # Save sparse matrix in current directory
        data_folder = './'

        sps.save_npz(os.path.join(data_folder, 'pro_words.npz'), self.tfidf_vectorizer)

        self.words_dict = dict({})
        for i in range(0, len(self.words_unique)):
            self.words_dict[str(i)] = self.words_unique[i]
        self.pro_num = dataframe_tf_idf.shape[0]
        self.words_num = dataframe_tf_idf.shape[1]

    def write_words_unique(self, data_folder):
        write_txt(os.path.join(data_folder, 'words_set.txt'), self.words_unique)

    def compute_similarity(self, shrink=10, topK=100, normalize=True, similarity="cosine"):
        self.shrink, self.topK, self.normalize, self.similarity = shrink, topK, normalize, similarity
        self.similarity_matrix = Compute_Similarity(self.tfidf_vectorizer.T, shrink=shrink, topK=topK,
                                                    normalize=normalize,
                                                    similarity=similarity).compute_similarity()

    def save_similarity_matrix(self, data_folder):
        sps.save_npz(os.path.join(data_folder, 'TF_IDF_pro_pro_' + str(self.shrink) + str(self.topK) + str(self.normalize) + '.npz'),
                    self.similarity_matrix)

    def _compute_problem_score(self, problems, corrects, target_problem):
        """

        """
        item_scores = self.similarity_matrix.tocsr()[problems, :].dot(
            self.similarity_matrix.tocsr().getrow(target_problem).transpose())
        item_scores = item_scores.transpose().todense().dot(corrects)
        if item_scores == 0.0:
            return -10.0
        return item_scores
