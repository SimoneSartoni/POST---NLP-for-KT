import pandas as pd
import numpy as np
import os
import scipy
from scipy import sparse as sps
from Knowledge_Tracing.code.Similarity.Compute_Similarity import Compute_Similarity

from sklearn.feature_extraction.text import TfidfVectorizer
from Knowledge_Tracing.code.models.base_model import base_model


def write_txt(file, data):
    with open(file, 'w') as f:
        for dd in data:
            f.write(str(dd) + '\n')


def identity_tokenizer(text):
    return text


class TF_IDF(base_model):
    def __init__(self):
        super().__init__("TF_IDF", "NLP")
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=identity_tokenizer,
            preprocessor=identity_tokenizer,
            token_pattern=None,
            use_idf=True)
        self.similarity_matrix = None
        self.words_unique = None
        self.pro_num = None
        self.words_num = None
        self.words_dict = None
        self.topK = 100
        self.shrink = 10
        self.normalize = True
        self.similarity = "cosine"
        self.vectors = None
        self.problem_id_to_index = {}
        self.problem_ids = None
        self.texts = None

    def fit(self, interacted_and_text_problems, problem_id_to_index, texts):
        self.problem_ids = interacted_and_text_problems
        self.texts = []
        index = 0
        for p in self.problem_ids:
            self.problem_id_to_index[p] = index
            self.texts.append(texts[problem_id_to_index[p]])
            index += 1
        tfidf_vectorizer_vectors = self.tfidf_vectorizer.fit_transform(self.texts)
        self.vectors = tfidf_vectorizer_vectors
        df_tf_idf = pd.DataFrame.sparse.from_spmatrix(tfidf_vectorizer_vectors)
        dataframe_tf_idf = df_tf_idf
        self.words_unique = self.tfidf_vectorizer.get_feature_names()
        # Save sparse matrix in current directory
        data_folder = './'

        sps.save_npz(os.path.join(data_folder, '../pro_words.npz'), tfidf_vectorizer_vectors)

        self.words_dict = {}
        for i in range(0, len(self.words_unique)):
            self.words_dict[str(i)] = self.words_unique[i]
        self.pro_num = dataframe_tf_idf.shape[0]
        self.words_num = dataframe_tf_idf.shape[1]

    def write_words_unique(self, data_folder):
        write_txt(os.path.join(data_folder, 'words_set.txt'), self.words_unique)

    def load_similarity_matrix(self, dataset_name):
        data_folder = "C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files/"
        self.similarity_matrix = sps.load_npz(os.path.join(data_folder, dataset_name + '/TF_IDF_similarity_' + str(self.shrink) + '_' + str(self.topK) + '_' + str(self.normalize) + '.npz'))

    def compute_similarity(self, shrink=10, topK=100, normalize=True, similarity="cosine", dataset_name=''):
        self.shrink, self.topK, self.normalize, self.similarity = shrink, topK, normalize, similarity
        self.similarity_matrix = Compute_Similarity(self.vectors.T, shrink=shrink, topK=topK,
                                                    normalize=normalize,
                                                    similarity=similarity).compute_similarity()
        self.save_similarity_matrix(dataset_name=dataset_name)

    def save_similarity_matrix(self, dataset_name):
        data_folder = "C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files/"
        sps.save_npz(os.path.join(data_folder,
                                  dataset_name+'/TF_IDF_similarity_' + str(self.shrink) + '_' + str(self.topK) + '_' + str(self.normalize) + '.npz'),
                     self.similarity_matrix)

    def compute_problem_score(self, input_problems, corrects, target_problem):
        """

        """
        input_ids = []
        correct_ids = []
        for p, c in list(zip(input_problems, corrects)):
            if p in self.problem_ids:
                # and p not in unique_problems_set:
                # unique_problems_set.add(p)
                input_ids.append(self.problem_id_to_index[p])
                correct_ids.append(c)
        item_scores = 0.0
        if target_problem in self.problem_ids:
            item_scores = self.similarity_matrix.tocsr()[input_ids, :].dot(
                self.similarity_matrix.tocsr().getrow(self.problem_id_to_index[target_problem]).transpose())
            item_scores = item_scores.transpose().todense().dot(correct_ids)
        return float(item_scores)
