from Knowledge_Tracing.code.experiments.modules.basic_experiment import basic_experiment
from Knowledge_Tracing.code.models.TF_IDF.TF_IDF import TF_IDF


class tf_idf(basic_experiment):
    def __init__(self, dataset, prediction_model):
        super().__init__()
        self.dataset = dataset
        self.encode_model = TF_IDF()
        self.prediction_model = prediction_model
        self.predictions = {}

    def encode(self):
        self.model.fit(self.dataset.interacted_with_text_problem_set, self.dataset.problem_id_to_index,
                       self.dataset.texts_list)

    def train(self, load=True, shrink=10, topK=100, normalize=True, similarity="cosine", dataset_name=''):
        if load:
            self.model.load_similarity_matrix(dataset_name=self.dataset.name)
        else:
            self.model.compute_similarity(shrink, topK, normalize, similarity, dataset_name)

    def compute_predictions(self):
        self.predictions = self.prediction_model.compute_predictions(self.dataset.test_set)
