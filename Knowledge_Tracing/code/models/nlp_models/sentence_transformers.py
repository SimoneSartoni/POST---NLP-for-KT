import gc

import pandas as pd
import numpy as np
import os
import scipy
from scipy import sparse as sps
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets, evaluation
from torch.utils.data import Dataset, DataLoader
import torch
from collections import Counter
        import math
from Knowledge_Tracing.code.Similarity.Compute_Similarity import Compute_Similarity
from Knowledge_Tracing.code.models.base_model import base_model


def write_txt(file, data):
    with open(file, 'w') as f:
        for dd in data:
            f.write(str(dd) + '\n')


def identity_tokenizer(text):
    return text


class SentenceSimilarityDataset(Dataset):
    def __init__(self, texts_df, text_column, frac=1):
        self.texts_df = texts_df.sample(frac=frac)
        self.texts_df_2 = texts_df.sample(frac=frac)
        print(self.texts_df)
        print(self.texts_df_2)
        self.text_column = text_column

    def __len__(self):
        return len(self.texts_df)

    def __getitem__(self, idx):
        texts_a = list(self.texts_df[self.text_column].values)[idx]
        texts_b = list(self.texts_df_2[self.text_column].values)[idx]
        list_a = list(self.texts_df_2['body'].values)[idx]
        list_b = list(self.texts_df_2['body'].values)[idx]
        counter_a = Counter(list_a)
        counter_b = Counter(list_b)
        def counter_cosine_similarity(c1, c2):
            terms = set(c1).union(c2)
            dot_product = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
            mag_a = math.sqrt(sum(c1.get(k, 0) ** 2 for k in terms))
            mag_b = math.sqrt(sum(c2.get(k, 0) ** 2 for k in terms))
            return dot_product / (mag_a * mag_b)
        cos_sim = counter_cosine_similarity(counter_a, counter_b)
        return InputExample(texts=[texts_a, texts_b], label=cos_sim)




class sentence_transformer(base_model):
    def __init__(self, encoding_model='all-mpnet-base-v2'):
        super().__init__("sentence_transformers", "NLP")
        self.st_model = SentenceTransformer(encoding_model)
        self.texts_df = None
        self.similarity_matrix = None
        self.words_unique = None
        self.pro_num = None
        self.words_num = None
        self.words_dict = None
        self.topK = 100
        self.shrink = 10
        self.normalize = True
        self.similarity = "cosine"
        self.vectors = {}
        self.vector_size = 0

    def fit_on_custom(self, texts_df, save_filepath='/content/', text_column="sentence", batch_size=128, frac=1):
        dataset = SentenceSimilarityDataset(texts_df, text_column, frac=frac)
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,)
        train_loss = losses.CosineSimilarityLoss(self.st_model)
        self.st_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=10, show_progress_bar=True)

    def transform(self, texts_df, text_coloumn='sentence', save_filepath='./'):
        self.texts_df = texts_df
        vectors = self.st_model.encode(sentences=self.texts_df[text_coloumn].values, show_progress_bar=True)
        for problem_id, encoding in list(zip(self.texts_df['problem_id'], vectors)):
            self.vectors[problem_id] = encoding
        del vectors
        gc.collect()
        # Save sparse matrix in current directory
        self.vector_size = len(list(self.vectors.values())[0])

        self.pro_num = len(list(self.vectors.values()))
        self.words_num = self.vector_size

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
            # and p not in unique_problems_set:
            # unique_problems_set.add(p)
            x = np.array(self.vectors[self.problem_id_to_index[p]])
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
        x = np.array(self.vectors[self.problem_id_to_index[target_problem]])
        target_encoding = target_encoding + x
        encoding = np.concatenate((pos_mean_encoding, neg_mean_encoding, target_encoding), axis=0)
        return encoding

    def get_encoding(self, problem_id):
        encoding = np.array(self.vectors[problem_id])
        return encoding

    def get_serializable_params(self):
        return {"min_df": self.min_df, "max_df": self.max_df, "binary":self.binary, "name": self.name, "topK": self.topK,
                "shrink": self.shrink, "normalize": self.normalize,
                "similarity": self.similarity}
