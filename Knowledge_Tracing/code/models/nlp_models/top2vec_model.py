import gc

import pandas as pd
import numpy as np
import os
import scipy
from scipy import sparse as sps
from Knowledge_Tracing.code.Similarity.Compute_Similarity import Compute_Similarity

from Knowledge_Tracing.code.models.base_model import base_model
from top2vec import Top2Vec

def write_txt(file, data):
    with open(file, 'w') as f:
        for dd in data:
            f.write(str(dd) + '\n')


def identity_tokenizer(text):
    return text


class Top2VecModel(base_model):
    def __init__(self, nr_topics=128,
                 calculate_probabilities=True, cluster_selection_method='eom'):
        super(Top2VecModel).__init__("Top2Vec", "NLP")
        self.topic_model = None
        self.probabilities = None
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
        self.vector_size = 0

    def fit(self, texts_df, save_filepath='./'):
        self.texts_df = texts_df
        self.topic_model = Top2Vec(documents=self.texts_df['sentence'].values,
                                   document_ids=self.texts_df['sentence'].keys,
                                   embedding_model='universal-sentence-encoder')
        print("topic model created")
        self.words_num = self.topic_model.get_num_topics()
        topic_sizes, topic_nums = self.topic_model.get_topic_sizes()
        print(topic_sizes)
        print(topic_nums)
        topic_words, word_scores, topic_scores, topic_nums = self.topic_model.search_topics(keywords=["algebra"], num_topics=5)
        for topic in topic_nums:
            self.topic_model.generate_topic_wordcloud(topic)

        documents, document_scores, document_ids = self.topic_model.search_documents_by_topic(topic_num=3, num_docs=5)
        for doc, score, doc_id in zip(documents, document_scores, document_ids):
            print(f"Document: {doc_id}, Score: {score}")
            print("-----------")
            print(doc)
            print("-----------")
            print()
        words, word_scores = self.topic_model.similar_words(keywords=["algebra"], keywords_neg=[], num_words=20)
        for word, score in zip(words, word_scores):
            print(f"{word} {score}")
        topic_predictions = self.topic_model.get_documents_topics(document_ids=self.texts_df['sentence'].
                                                                                 keys)

        gc.collect()
        topic_predictions = self.topic_model.get_documents_topics(document_ids=self.texts_df['sentence'].
                                                                  keys)

        vectors = self.topic_model._get_document_vectors()
        self.vectors = {}
        for problem_id, vector in list(zip(self.texts_df['problem_id'], vectors)):
            self.vectors[problem_id] = vector
        del vectors
        gc.collect()
        self.texts_df['topics'] = topic_predictions
        self.vector_size = self.words_num
        self.pro_num = len(self.texts_df['sentence'].values)

    def write_words_unique(self, data_folder):
        write_txt(os.path.join(data_folder, 'words_set.txt'), self.words_unique)

    def load_similarity_matrix(self, dataset_name):
        data_folder = "C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files/"
        self.similarity_matrix = sps.load_npz(os.path.join(data_folder, dataset_name + '/TF_IDF_similarity_' + str(
            self.shrink) + '_' + str(self.topK) + '_' + str(self.normalize) + '.npz'))

    def compute_similarity(self, shrink=10, topK=100, normalize=True, similarity="cosine", dataset_name='',
                           dataset_prefix=''):
        self.shrink, self.topK, self.normalize, self.similarity = shrink, topK, normalize, similarity
        self.similarity_matrix = Compute_Similarity(self.vectors.T, shrink=shrink, topK=topK,
                                                    normalize=normalize,
                                                    similarity=similarity).compute_similarity()
        self.save_similarity_matrix(name=dataset_name, prefix=dataset_prefix)

    def save_similarity_matrix(self, name, prefix):
        data_folder = "C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files/"
        path = os.path.join(data_folder, prefix)
        path = os.path.join(path, name + "/")
        path = os.path.join(path, 'TF_IDF_similarity_' + str(self.shrink) + '_' + str(self.topK) + '_' +
                            str(self.normalize) + '.npz')
        sps.save_npz(path, self.similarity_matrix)

    def compute_problem_score(self, input_problems, corrects, target_problem):
        item_scores, corrects = self.compute_similarities(input_problems, corrects, target_problem)
        item_scores = np.array(item_scores).dot(corrects)
        return float(item_scores)

    def compute_similarities(self, input_problems, corrects, target_problem):
        input_ids = []
        correct_ids = []
        for p, c in list(zip(input_problems, corrects)):
            if p in self.problem_id_to_index.keys():
                # and p not in unique_problems_set:
                # unique_problems_set.add(p)
                input_ids.append(self.problem_id_to_index[p])
                correct_ids.append(c)
        if len(input_problems) == 0:
            return [0.0], [0.0]
        if target_problem in self.problem_id_to_index.keys():
            item_scores = self.similarity_matrix.tocsr()[input_ids, :].dot(
                self.similarity_matrix.tocsr().getrow(self.problem_id_to_index[target_problem]).transpose())
            item_scores = item_scores.transpose().todense()
        else:
            return [0.0], [0.0]
        return item_scores, correct_ids

    def get_encoding(self, problem_id):
        encodings = np.array(self.vectors[problem_id])
        return encodings

    def get_serializable_params(self):
        return {"min_df": self.min_df, "max_df": self.max_df, "binary": self.binary, "name": self.name,
                "topK": self.topK,
                "shrink": self.shrink, "normalize": self.normalize,
                "similarity": self.similarity}
