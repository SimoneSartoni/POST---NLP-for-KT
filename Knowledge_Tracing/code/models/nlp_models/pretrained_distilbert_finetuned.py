import gc
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os
import scipy
from scipy import sparse as sps
from Knowledge_Tracing.code.Similarity.Compute_Similarity import Compute_Similarity
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertModel
from sentence_transformers import InputExample, models, SentenceTransformer
import datasets
from Knowledge_Tracing.code.models.base_model import base_model
import torch
from torch.utils.data import Dataset, DataLoader



def write_txt(file, data):
    with open(file, 'w') as f:
        for dd in data:
            f.write(str(dd) + '\n')


def identity_tokenizer(text):
    return text


# define mean pooling function
def mean_pool(token_embeds, attention_mask):
    # reshape attention_mask to cover 768-dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()
    # perform mean-pooling but exclude padding tokens (specified by in_mask)
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
    return pool


class SentenceSimilarityDataset(Dataset):
    def __init__(self, texts_df, text_column):
        self.texts_df = texts_df
        self.texts_df_2 = texts_df.sample(frac=1)
        self.text_column = text_column

    def __len__(self):
        return len(self.texts_df)

    def __getitem__(self, idx, ):
        print(self.texts_df[self.text_column].values)
        texts = self.texts_df[self.text_column].values[idx]
        texts_2 = self.texts_df_2[self.text_column].values[idx]
        anchor_ids, anchor_mask = self.tokenizer(
            texts, max_length=128, padding='max_length',
            truncation=True
        )
        print(anchor_ids)
        print(anchor_mask)
        positive_ids, positive_mask = self.tokenizer(
            texts_2, max_length=128, padding='max_length',
            truncation=True
        )
        inputs = {"anchor_ids": anchor_ids, "anchor_mask": anchor_mask,
                  "positive_ids": positive_ids, "positive_mask": positive_mask}
        return inputs


class PretrainedDistilBERTFinetuned(base_model):
    def __init__(self, config_path="/content/drive/MyDrive/simone sartoni - text enhanced deep knowledge tracing/"
                                   "pretrained_distilbert_base_uncased_24_epochs/config.json",
                 model_filepath="/content/drive/MyDrive/simone sartoni - text enhanced deep knowledge tracing/"
                                "pretrained_distilbert_base_uncased_24_epochs/tf_model.h5"):
        super().__init__("PretrainedDistilBERT", "NLP")

        """
        self.embeddings = None
        self.umap_embeddings = None
        self.umap = umap.UMAP(n_neighbors=n_neighbors,
                            n_components=n_components,
                            metric=metric)
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                  metric=metric,
                                  cluster_selection_method=cluster_selection_method)
        """

        self.config = DistilBertConfig.from_json_file(config_path)
        self.model = DistilBertModel.from_pretrained(model_filepath, config=self.config, from_tf=True)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.sentence_model = None
        self.encodings = {}
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

    def fit_on_CA(self, texts_df, save_filepath='/content/', text_column="sentence"):
        self.texts_df = texts_df
        dataset = SentenceSimilarityDataset(texts_df=texts_df, text_column=text_column)
        batch_size = 32

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        # set device and move model there
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        print(f'moved to {device}')
        # define layers to be used in multiple-negatives-ranking
        cos_sim = torch.nn.CosineSimilarity()
        loss_func = torch.nn.CrossEntropyLoss()
        scale = 20.0  # we multiply similarity score by this scale value
        # move layers to device
        cos_sim.to(device)
        loss_func.to(device)
        from transformers.optimization import get_linear_schedule_with_warmup
        # initialize Adam optimizer
        optim = torch.optim.Adam(self.model.parameters(), lr=2e-5)
        # setup warmup for first ~10% of steps
        total_steps = int(len(dataset) / batch_size)
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optim, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps - warmup_steps
        )
        from tqdm.auto import tqdm
        epochs = 1
        # 1 epoch should be enough, increase if wanted
        for epoch in range(epochs):
            self.model.train()  # make sure model is in training mode
            # initialize the dataloader loop with tqdm (tqdm == progress bar)
            loop = tqdm(loader, leave=True)
            for batch in loop:
                # zero all gradients on each new step
                optim.zero_grad()
                # prepare batches and more all to the active device
                anchor_ids = batch['anchor_ids'].to(device)
                anchor_mask = batch['anchor_mask'].to(device)
                pos_ids = batch['positive_ids'].to(device)
                pos_mask = batch['positive_mask'].to(device)
                # extract token embeddings from BERT
                a = self.model(
                    anchor_ids, attention_mask=anchor_mask
                )[0]  # all token embeddings
                p = self.model(
                    pos_ids, attention_mask=pos_mask
                )[0]
                # get the mean pooled vectors
                a = mean_pool(a, anchor_mask)
                p = mean_pool(p, pos_mask)
                # calculate the cosine similarities
                scores = torch.stack([
                    cos_sim(
                        a_i.reshape(1, a_i.shape[0]), p
                    ) for a_i in a])
                # get label(s) - we could define this before if confident of consistent batch sizes
                labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
                # and now calculate the loss
                loss = loss_func(scores * scale, labels)
                # using loss, calculate gradients and then optimize
                loss.backward()
                optim.step()
                # update learning rate scheduler
                scheduler.step()
                # update the TDQM progress bar
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())

        import os
        model_path = '/content/sbert_trained_on_CA/'
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        self.model.save_pretrained(model_path)

    def fit_on_nli(self, save_filepath='/content/', ):
        snli = datasets.load_dataset('snli', split='train')
        mnli = datasets.load_dataset('glue', 'mnli', split='train')
        mnli = mnli.remove_columns("idx")
        print(mnli)
        print(snli)
        snli = snli.cast(mnli.features)
        dataset = datasets.concatenate_datasets([snli, mnli])
        del snli, mnli

        print(f"before: {len(dataset)} rows")
        dataset = dataset.filter(
            lambda x: True if x['label'] == 0 else False
        )
        print(f"after: {len(dataset)} rows")

        dataset = dataset.map(
            lambda x: self.tokenizer(
                x['premise'], max_length=128, padding='max_length',
                truncation=True
            ), batched=True
        )
        dataset = dataset.rename_column('input_ids', 'anchor_ids')
        dataset = dataset.rename_column('attention_mask', 'anchor_mask')
        dataset = dataset.map(
            lambda x: self.tokenizer(
                x['hypothesis'], max_length=128, padding='max_length',
                truncation=True
            ), batched=True
        )
        dataset = dataset.rename_column('input_ids', 'positive_ids')
        dataset = dataset.rename_column('attention_mask', 'positive_mask')
        dataset = dataset.remove_columns(['premise', 'hypothesis', 'label'])
        dataset.set_format(type='torch', output_all_columns=True)
        batch_size = 32

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        # set device and move model there
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        print(f'moved to {device}')
        # define layers to be used in multiple-negatives-ranking
        cos_sim = torch.nn.CosineSimilarity()
        loss_func = torch.nn.CrossEntropyLoss()
        scale = 20.0  # we multiply similarity score by this scale value
        # move layers to device
        cos_sim.to(device)
        loss_func.to(device)
        from transformers.optimization import get_linear_schedule_with_warmup
        # initialize Adam optimizer
        optim = torch.optim.Adam(self.model.parameters(), lr=2e-5)
        # setup warmup for first ~10% of steps
        total_steps = int(len(dataset) / batch_size)
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optim, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps - warmup_steps
        )
        from tqdm.auto import tqdm
        epochs = 1
        # 1 epoch should be enough, increase if wanted
        for epoch in range(epochs):
            self.model.train()  # make sure model is in training mode
            # initialize the dataloader loop with tqdm (tqdm == progress bar)
            loop = tqdm(loader, leave=True)
            for batch in loop:
                # zero all gradients on each new step
                optim.zero_grad()
                # prepare batches and more all to the active device
                anchor_ids = batch['anchor_ids'].to(device)
                anchor_mask = batch['anchor_mask'].to(device)
                pos_ids = batch['positive_ids'].to(device)
                pos_mask = batch['positive_mask'].to(device)
                # extract token embeddings from BERT
                a = self.model(
                    anchor_ids, attention_mask=anchor_mask
                )[0]  # all token embeddings
                p = self.model(
                    pos_ids, attention_mask=pos_mask
                )[0]
                # get the mean pooled vectors
                a = mean_pool(a, anchor_mask)
                p = mean_pool(p, pos_mask)
                # calculate the cosine similarities
                scores = torch.stack([
                    cos_sim(
                        a_i.reshape(1, a_i.shape[0]), p
                    ) for a_i in a])
                # get label(s) - we could define this before if confident of consistent batch sizes
                labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
                # and now calculate the loss
                loss = loss_func(scores * scale, labels)
                # using loss, calculate gradients and then optimize
                loss.backward()
                optim.step()
                # update learning rate scheduler
                scheduler.step()
                # update the TDQM progress bar
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())

        import os
        model_path = '/content/pretrained_sbert_mnr'
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        self.model.save_pretrained(model_path)

    def fit(self, texts_df, save_filepath="", text_column="sentence"):
        self.texts_df = texts_df
        start = 0
        batch_size = 100
        while start < len(texts_df.index):
            end = start + batch_size
            inputs = self.tokenizer(list(self.texts_df[text_column].values)[start:end], truncation=True,
                                    return_tensors="tf",
                                    padding=True)
            attention_mask = inputs['attention_mask'].numpy()
            output = self.model(inputs)
            encoding = output.to_tuple()[0].numpy()
            for problem_id, enc, attention in list(zip(self.texts_df['problem_id'].values[start:end],
                                                       encoding, attention_mask)):
                self.encodings[problem_id] = mean_pool(enc, attention)
            start = start + batch_size
        # Save sparse matrix in current directory
        self.vector_size = len(list(self.vectors.values())[0])

        self.pro_num = len(list(self.vectors.values()))
        self.words_num = self.vector_size
        print(len(list(self.encodings.keys())))
        print("pretrainedBERT model created")

        self.words_num = list(self.encodings.values())[0].shape[0]
        print("vector_size: " + str(self.words_num))
        self.vector_size = self.words_num
        self.pro_num = len(self.texts_df.index)

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
        encodings = np.array(self.encodings[problem_id])
        return encodings

    def get_serializable_params(self):
        return {"min_df": self.min_df, "max_df": self.max_df, "binary": self.binary, "name": self.name,
                "topK": self.topK,
                "shrink": self.shrink, "normalize": self.normalize,
                "similarity": self.similarity}
