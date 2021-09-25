from abc import ABC

from Knowledge_Tracing.code.experiments.modules.basic_experiment import basic_experiment
from Knowledge_Tracing.code.models.gensim_model.gensim_word2vec import world2vec


class word2vec(basic_experiment):
    def __init__(self, dataset, prediction_model, load=False, min_count=2, window=5, vector_size=100, workers=3, sg=1, epochs=20):
        super().__init__(name="word2vec_"+dataset.name+"_"+prediction_model.name)
        self.dataset = dataset
        self.encode_model = None
        self.load = load
        self.epochs = epochs
        self.prediction_model = prediction_model
        self.predictions = []
        self.labels = []
        self.min_count = min_count
        self.window = window
        self.vector_size = vector_size
        self.sg = sg
        self.workers = workers
        self.epochs = epochs

    def encode(self):
        self.encode_model = world2vec()
        if not self.load:
            self.encode_model.fit(self.dataset.texts_list, epochs=self.epochs)
        else:
            self.encode_model.load_model(epochs=self.epochs,
                                         name=self.dataset.name)
            self.encode_model.load_word_vectors(epochs=self.epochs,
                                                name=self.dataset.name)
        self.encode_model.encode_problems(self.dataset.problem_id_to_index, self.dataset.texts_list)

    def prediction_train(self):
        self.prediction_model.train(self.encode_model, self.dataset.train_set)

    def compute_predictions(self):
        self.labels, self.predictions = self.prediction_model.compute_predictions(self.dataset.test_set)

    def set_params(self, **parameters):
        for parameter, value in list(parameters.items()):
            setattr(self, parameter, value)
        self.encode_model = world2vec(name="word2vec_size" + str(self.vector_size) + "_epoch" + str(self.epochs),
                                      class_of_method="NLP", min_count=self.min_count, window=self.window,
                                      vector_size=self.vector_size, workers=self.workers, sg=self.sg)
        return self
