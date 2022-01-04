import gc
import numpy as np
from bertopic import BERTopic
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertModel, TFDistilBertModel
from sentence_transformers import SentenceTransformer, models


class BERTopic_model():
    def __init__(self, nr_topics=128, calculate_probabilities=True, cluster_selection_method='eom', output="probability"):
        """self.sentence_transformer = SentenceTransformer('bert-large-nli-mean-tokens')
        self.embeddings = None
        self.umap_embeddings = None
        self.umap = umap.UMAP(n_neighbors=n_neighbors,
                            n_components=n_components,
                            metric=metric)
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                  metric=metric,
                                  cluster_selection_method=cluster_selection_method)"""
        self.name = "BERTopic"
        self.method = "NLP"
        self.nr_topics = nr_topics
        self.calculate_probabilities = calculate_probabilities
        self.cluster_selection_method = cluster_selection_method
        self.bertopic = None
        self.word_embedding_model = None
        self.st_model = None
        self.tokenizer = None
        self.topic_model = None
        self.probabilities = {}
        self.topics = {}
        self.pro_num = 0
        self.words_num = 0
        self.topic_names = None
        self.vector_size = 0
        self.output = output
        self.texts_df = None

    def initialize_pretrained_bertopic(self, model_path_or_name="all-mpnet-base-v2"):
        self.st_model = SentenceTransformer(model_path_or_name)
        self.bertopic = BERTopic(embedding_model=model_path_or_name, language="english",
                                 calculate_probabilities=self.calculate_probabilities, nr_topics=self.nr_topics)

    def initialize_custom_bertopic(self, config_path="", model_path_or_name="", tokenizer_name="distilbert-base-uncased"
                                   , from_tf=True):
        config = DistilBertConfig.from_json_file(config_path)
        self.word_embedding_model = DistilBertModel.from_pretrained(model_path_or_name, config=config, from_tf=from_tf)
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        pooling_model = models.Pooling(768)
        self.st_model = SentenceTransformer(modules=[self.word_embedding_model, pooling_model])
        self.bertopic = BERTopic(embedding_model=self.st_model, language="english",
                                 calculate_probabilities=self.calculate_probabilities, nr_topics=self.nr_topics)

    def fit(self, texts_df, text_column="sentence"):
        self.topic_model = self.bertopic.fit(texts_df['sentence'].values)
        print("topic model created")
        self.words_num = len(self.topic_model.get_topic_freq())
        self.topic_names = self.topic_model.get_topics().keys()
        print(self.topic_names)
        self.vector_size = len(self.topic_names) - 1

    def transform(self, texts_df, save_filepath='/content/'):
        self.texts_df = texts_df
        self.pro_num = len(list(texts_df.index))
        topic_predictions, probabilities = self.topic_model.transform(texts_df['sentence'].values)
        self.probabilities = {}
        self.topics = {}
        for problem_id, probability, topic_prediction in list(zip(self.texts_df['problem_id'], probabilities, topic_predictions)):
            self.probabilities[problem_id] = probability
            self.topics[problem_id] = topic_prediction
        del probabilities
        gc.collect()
        self.texts_df['topics'] = topic_predictions
        self.texts_df.to_csv(save_filepath + 'text_df_with_topics.csv')
        self.topic_model.visualize_topics()
        self.topic_model.visualize_hierarchy(topics=range(0, 10))
        self.topic_model.visualize_barchart(topics=range(0, 10))
        self.topic_model.visualize_heatmap(topics=range(0, 10))
        self.topic_model.visualize_term_rank()

    def get_encoding(self, problem_id):
        encodings = np.array([])
        if self.output == "probability":
            encodings = np.array(self.probabilities[problem_id])
        if self.output == "topic":
            encodings = np.array(self.topics[problem_id])
        return encodings
