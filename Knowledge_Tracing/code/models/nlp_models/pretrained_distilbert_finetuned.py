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
from tqdm.auto import tqdm  # so we see progress bar
import datasets
from Knowledge_Tracing.code.models.base_model import base_model


def write_txt(file, data):
    with open(file, 'w') as f:
        for dd in data:
            f.write(str(dd) + '\n')


def identity_tokenizer(text):
    return text


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), token_embeddings.shape)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, 0)
    sum_mask = np.clip(np.sum(input_mask_expanded, 0), 1e-9, 1000)
    return sum_embeddings / sum_mask


# Max Pooling - Take attention mask into account for correct max
def max_pooling(token_embeddings, attention_mask):
    input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), token_embeddings.shape)
    max_embeddings = np.amax(token_embeddings * input_mask_expanded, 0)
    return max_embeddings


# Min Pooling - Take attention mask into account for correct averaging
def min_pooling(token_embeddings, attention_mask):
    input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), token_embeddings.shape)
    min_embeddings = np.amin(token_embeddings * input_mask_expanded, 0)
    return min_embeddings


class PretrainedDistilBERT(base_model):
    def __init__(self, config_path="/content/drive/MyDrive/simone sartoni - text enhanced deep knowledge tracing/"
                                   "pretrained_distilbert_base_uncased_24_epochs/config.json",
                 model_filepath="/content/drive/MyDrive/simone sartoni - text enhanced deep knowledge tracing/"
                                "pretrained_distilbert_base_uncased_24_epochs/tf_model.h5",
                 pooling='mean'
                 ):
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
        self.model = DistilBertModel.from_pretrained(model_filepath, config=self.config)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.sentence_model = None
        self.encodings = {}
        if pooling == 'mean':
            self.pooling_method = mean_pooling
        elif pooling == 'min':
            self.pooling_method = min_pooling
        elif pooling == 'max':
            self.pooling_method = max_pooling
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

    def fit(self, texts_df, save_filepath='./', text_coloumn="sentence"):
        self.texts_df = texts_df
        start = 0
        batch_size = 100
        snli = datasets.load_dataset('snli', split='train')
        mnli = datasets.load_dataset('glue', 'mnli', split='train')
        snli = snli.cast(mnli.features)
        dataset = datasets.concatenate_datasets([snli, mnli])
        del snli, mnli

        print(f"before: {len(dataset)} rows")
        dataset = dataset.filter(
            lambda x: True if x['label'] == 0 else False
        )
        print(f"after: {len(dataset)} rows")

        train_samples = []
        for row in tqdm(dataset):
            train_samples.append(InputExample(
                texts=[row['premise'], row['hypothesis']]
            ))

        from sentence_transformers import datasets as datasets_2

        batch_size = 32

        loader = datasets_2.NoDuplicatesDataLoader(train_samples, batch_size=batch_size)
        pooler = models.Pooling(
            self.model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )

        self.sentence_model = SentenceTransformer(modules=[self.model, pooler])
        from sentence_transformers import losses
        loss = losses.MultipleNegativesRankingLoss(self.sentence_model)

        epochs = 1
        warmup_steps = int(len(loader) * epochs * 0.1)
        self.sentence_model.fit(
            train_objectives=[(loader, loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path='/content/sbert_test_mnr2',
            show_progress_bar=False
        )  # I set 'show_progress_bar=False' as it printed every step
        #    on to a new line

        vectors = self.sentence_model.encode(sentences=self.texts_df[text_coloumn].values, show_progress_bar=True)
        for problem_id, encoding in list(zip(self.texts_df['problem_id'], vectors)):
            self.vectors[problem_id] = encoding
        del vectors
        gc.collect()
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
