from Knowledge_Tracing.code.experiments.modules.basic_experiment import basic_experiment
from Knowledge_Tracing.code.models.count_vectorizer.count_vectorizer import count_vectorizer
from mlflow import *


class count_vect(basic_experiment):
    def __init__(self, dataset, prediction_model, min_df=1, max_df=1.0, binary=False, load=False, shrink=10, topK=100, normalize=True,
                 similarity="cosine"):
        super().__init__(name="count_vect_" + dataset.name + "_" + prediction_model.name + "_min_df_" + str(min_df))
        self.dataset = dataset
        self.encode_model = None
        self.prediction_model = prediction_model
        self.predictions = []
        self.labels = []

        self.min_df = min_df
        self.max_df = max_df
        self.binary = binary
        self.load = load
        self.shrink = shrink
        self.topK = topK
        self.normalize = normalize
        self.similarity = similarity

    def encode(self):
        self.encode_model = count_vectorizer(min_df=self.min_df, max_df=self.max_df, binary=self.binary)
        self.encode_model.fit(self.dataset.interacted_with_text_problem_set, self.dataset.problem_id_to_index,
                              self.dataset.texts_list)
        if self.load:
            self.encode_model.load_similarity_matrix(dataset_name=self.dataset.name)
        else:
            self.encode_model.compute_similarity(self.shrink, self.topK, self.normalize, self.similarity,
                                                 dataset_name=self.dataset.name, dataset_prefix=self.dataset.prefix)

    def prediction_train(self):
        self.prediction_model.train(self.encode_model, self.dataset.train_set)

    def compute_predictions(self):
        self.labels, self.predictions = self.prediction_model.compute_predictions(self.dataset.test_set)

    def fit(self):
        self.encode()
        self.prediction_train()

    def predict(self, X=None):
        self.compute_predictions()
        return self.predictions

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"dataset": self.dataset, "prediction_model": self.prediction_model, "topK": self.topK, "shrink": self.shrink, "normalize": self.normalize, "similarity": self.similarity}

    def set_params(self, **parameters):
        for parameter, value in list(parameters.items()):
            setattr(self, parameter, value)
        return self
