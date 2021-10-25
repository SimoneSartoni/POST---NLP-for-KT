import pandas as pd
import numpy as np
import os
import scipy
from scipy import sparse as sps
from sentence_transformers import SentenceTransformer
from Knowledge_Tracing.code.Similarity.Compute_Similarity import Compute_Similarity

from sklearn.feature_extraction.text import CountVectorizer
from Knowledge_Tracing.code.models.base_model import base_model
from bertopic import BERTopic
import hdbscan


def write_txt(file, data):
    with open(file, 'w') as f:
        for dd in data:
            f.write(str(dd) + '\n')


def identity_tokenizer(text):
    return text


class BERTopic_model(base_model):
    def __init__(self, n_neighbors=15, n_components=5, metric='cosine', min_cluster_size=15, metric='euclidean',
                 cluster_selection_method='eom'):
        super().__init__("sentence_transformers", "NLP")
        """self.sentence_transformer = SentenceTransformer('bert-large-nli-mean-tokens')
        self.embeddings = None
        self.umap_embeddings = None
        self.umap = umap.UMAP(n_neighbors=n_neighbors,
                            n_components=n_components,
                            metric=metric)
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                  metric=metric,
                                  cluster_selection_method=cluster_selection_method)"""
        self.bert_topic = BERTopic(embedding_model='bert-large-nli-mean-tokens', language="english")
        self.topics, self.probabilities = None, None
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

    def fit(self, interacted_and_text_problems, problem_id_to_index, texts, save_filepath='./'):
        self.problem_ids = interacted_and_text_problems
        self.texts = []
        index = 0
        for p in self.problem_ids:
            self.problem_id_to_index[p] = index
            self.texts.append(texts[problem_id_to_index[p]])
            index += 1
        self.topics, self.probabilities = self.bert_topic.fit_transform(self.texts)
        print(self.topics[0:100])
        print(self.probabilities[0:100])
        # Save sparse matrix in current directory
        self.vector_size = self.vectors.shape[1]

        sps.save_npz(os.path.join(save_filepath, '../pro_words.npz'), self.vectors)

        self.pro_num = self.vectors.shape[0]
        self.words_num = self.vectors.shape[1]

    def visualize_umap():
        result = pd.DataFrame(umap_data, columns=['x', 'y'])
        result['labels'] = cluster.labels
        # Visualize clusters
        fig, ax = plt.subplots(figsize=(20, 10))
        outliers = result.loc[result.labels == -1, :]
        clustered = result.loc[result.labels != -1, :]
        plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
        plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
        plt.colorbar()

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
        item_scores = 0.0
        if len(input_problems) == 0:
            return [0.0], [0.0]
        if target_problem in self.problem_id_to_index.keys():
            item_scores = self.similarity_matrix.tocsr()[input_ids, :].dot(
                self.similarity_matrix.tocsr().getrow(self.problem_id_to_index[target_problem]).transpose())
            item_scores = item_scores.transpose().todense()
        else:
            return [0.0], [0.0]
        return item_scores, correct_ids

    def compute_encoding(self, input_problems, corrects, target_problem):
        pos_mean_encoding = np.zeros(shape=self.words_num, dtype=np.float)
        neg_mean_encoding = np.zeros(shape=self.words_num, dtype=np.float)
        pos, neg = 0.0, 0.0
        for p, c in list(zip(input_problems, corrects)):
            if p in self.problem_id_to_index.keys():
                # and p not in unique_problems_set:
                # unique_problems_set.add(p)
                problem = self.problem_id_to_index[p]
                x = np.array(self.vectors[problem])
                if c > 0.0:
                    pos += 1.0
                    pos_mean_encoding = pos_mean_encoding + x
                else:
                    neg += 1.0
                    neg_mean_encoding = neg_mean_encoding + x
        if pos > 0.0:
            pos_mean_encoding = pos_mean_encoding / pos
        if neg > 0.0:
            neg_mean_encoding = neg_mean_encoding / neg
        target_encoding = np.zeros(shape=self.words_num, dtype=np.float)
        if target_problem in self.problem_id_to_index.keys():
            x = np.array(self.vectors[self.problem_id_to_index[target_problem]])
            target_encoding = target_encoding + x
        encoding = np.concatenate((pos_mean_encoding, neg_mean_encoding, target_encoding), axis=0)
        return encoding

    def get_encoding(self, problem):
        index = self.problem_id_to_index[problem]
        encoding = np.array(self.vectors[index])
        return encoding

    def get_serializable_params(self):
        return {"min_df": self.min_df, "max_df": self.max_df, "binary":self.binary, "name": self.name, "topK": self.topK,
                "shrink": self.shrink, "normalize": self.normalize,
                "similarity": self.similarity}
