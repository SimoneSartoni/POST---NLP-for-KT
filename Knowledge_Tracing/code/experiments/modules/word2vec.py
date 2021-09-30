from abc import ABC

from Knowledge_Tracing.code.experiments.modules.basic_experiment import basic_experiment
from Knowledge_Tracing.code.models.gensim_model.gensim_word2vec import world2vec
from Knowledge_Tracing.code.models.gensim_model.gensim_pretrained_word2vec import pretrained_world2vec


class word2vec(basic_experiment):
    def __init__(self, dataset, encode_model, prediction_model):
        super().__init__(name="word2vec_"+dataset.name+"_"+prediction_model.name)
        self.dataset = dataset
        self.encode_model = encode_model
        self.prediction_model = prediction_model
        self.predictions = []
        self.labels = []

    def encode(self):
        self.encode_model.fit(self.dataset.texts_list)
        self.encode_model.encode_problems(self.dataset.problem_id_to_index, self.dataset.texts_list)

    def prediction_train(self):
        self.prediction_model.train(self.encode_model, self.dataset.train_set)

    def compute_predictions(self):
        self.labels, self.predictions = self.prediction_model.compute_predictions(self.dataset.test_set)

    def set_params(self, **parameters):
        for parameter, value in list(parameters.items()):
            setattr(self, parameter, value)
        return self
