from Knowledge_Tracing.code.experiments.modules.basic_experiment import basic_experiment
from Knowledge_Tracing.code.models.TF_IDF.TF_IDF import TF_IDF
from mlflow import *

class tf_idf(basic_experiment):
    def __init__(self, dataset, prediction_model, load=True, shrink=10, topK=100, normalize=True, similarity="cosine"):
        super().__init__(name="tf_idf_"+dataset.name+"_"+prediction_model.name)
        self.dataset = dataset
        self.encode_model = TF_IDF()
        self.prediction_model = prediction_model
        self.predictions = []
        self.labels = []

        self.load = load
        self.shrink = shrink
        self.topK = topK
        self.normalize = normalize
        self.similarity = similarity

    def encode(self):
        self.encode_model.fit(self.dataset.interacted_with_text_problem_set, self.dataset.problem_id_to_index,
                       self.dataset.texts_list)
        if self.load:
            self.encode_model.load_similarity_matrix(dataset_name=self.dataset.name)
        else:
            self.encode_model.compute_similarity(self.shrink, self.topK, self.normalize, self.similarity, self.dataset.name)

    def prediction_train(self):
        self.prediction_model.train(self.encode_model, self.dataset.train_set, self.dataset.validation_set)

    def compute_predictions(self):
        self.labels, self.predictions = self.prediction_model.compute_predictions(self.dataset.test_set)
        self.save_predictions()
