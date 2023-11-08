import gc

import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import Dataset
import torch
import math


def write_txt(file, data):
    with open(file, 'w') as f:
        for dd in data:
            f.write(str(dd) + '\n')


def identity_tokenizer(text):
    return text


class SentenceSimilarityDataset(Dataset):
    def __init__(self, texts_df, text_column, frac=1):
        texts_df[text_column].fillna("na", inplace=True)
        self.texts_df = texts_df.sample(frac=frac, random_state=1)
        self.texts_df_2 = texts_df.sample(frac=frac, random_state=10)
        print(self.texts_df)
        print(self.texts_df_2)
        self.text_column = text_column

    def __len__(self):
        return len(self.texts_df)

    def __getitem__(self, idx):
        texts_a = list(self.texts_df[self.text_column].values)[idx]
        texts_b = list(self.texts_df_2[self.text_column].values)[idx]
        list_a = list(self.texts_df['list_of_words'].values)[idx]
        list_b = list(self.texts_df_2['list_of_words'].values)[idx]
        if not texts_a:
            texts_a == "not available"
        if not texts_b:
            texts_b == "not available"

        def counter_cosine_similarity(c1, c2):
            common = set(c1).intersection(set(c2))
            dot_product = len(common)
            mag_a = math.sqrt(float(len(set(c1))))
            mag_b = math.sqrt(float(len(set(c2))))
            if mag_a * mag_b != 0:
                return float(dot_product) / float(mag_a * mag_b)
            else:
                return 0.0
        cos_sim = counter_cosine_similarity(list_a, list_b)
        return InputExample(texts=[texts_a, texts_b], label=cos_sim)


class sentence_transformer:
    def __init__(self, encoding_model='all-mpnet-base-v2'):
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
        self.embeddings = {}
        self.vector_size = 0
        self.name = "sentence_transformer"

    def fit_on_custom(self, texts_df, save_filepath='/content/', text_column="sentence", batch_size=128, frac=1,
                      epochs=1):
        dataset = SentenceSimilarityDataset(texts_df, text_column, frac=frac)
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, )
        train_loss = losses.CosineSimilarityLoss(self.st_model)
        self.st_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, warmup_steps=10,
                          show_progress_bar=True)

    def transform(self, texts_df, text_column='sentence', save_filepath='./'):
        self.texts_df = texts_df
        self.texts_df[text_column].fillna("na", inplace=True)
        length = len(self.texts_df[text_column].values)

        embeddings = self.st_model.encode(sentences=self.texts_df[text_column].values[0:length // 2],
                                          show_progress_bar=True)
        for problem_id, embedding in list(zip(list(self.texts_df['problem_id'].values[0:length // 2]), embeddings)):
            self.embeddings[problem_id] = embedding
        del embeddings
        gc.collect()
        embeddings = self.st_model.encode(sentences=self.texts_df[text_column].values[length // 2:],
                                          show_progress_bar=True)
        for problem_id, embedding in list(zip(list(self.texts_df['problem_id'].values[length // 2:]), embeddings)):
            self.embeddings[problem_id] = embedding
        del embeddings
        gc.collect()
        np.save(save_filepath+"st_embeddings.npy", self.embeddings)

        # Save sparse matrix in current directory
        self.vector_size = len(list(self.embeddings.values())[0])

        self.pro_num = len(list(self.embeddings.values()))
        self.words_num = self.vector_size

    def load_embeddings(self, load_path=""):
        self.embeddings = np.load(load_path, allow_pickle=True).item()
        #print(self.embeddings)
        self.vector_size = len(list(self.embeddings.values())[0])

        self.pro_num = len(list(self.embeddings.values()))
        self.words_num = self.vector_size

    def get_encoding(self, problem_id):
        encoding = np.array(self.embeddings[problem_id])
        return encoding
