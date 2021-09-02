from abc import ABC

from Knowledge_Tracing.code.experiments.modules.basic_experiment import basic_experiment
from Knowledge_Tracing.code.models.TF_IDF.TF_IDF import TF_IDF
from Knowledge_Tracing.code.models.gensim_model.gensim_word2vec import world2vec


class word2vec(basic_experiment):
    def __init__(self, dataset, prediction_model, load=False, min_count=2, window=5, vector_size=100, workers=3, sg=1, epochs=20):
        super().__init__(name="word2vec_"+dataset.name+"_"+prediction_model.name)
        self.dataset = dataset
        self.encode_model = world2vec(name="word2vec_size" + str(vector_size) + "_epoch" + str(epochs),
                                      class_of_method="NLP",
                                      min_count=min_count, window=window, vector_size=vector_size, workers=workers,
                                      sg=sg)
        self.load = load
        self.epochs = epochs
        self.prediction_model = prediction_model
        self.predictions = []
        self.labels = []

    def encode(self):
        if not self.load:
            self.encode_model.fit(self.dataset.texts_list, epochs=self.epochs)
        else:
            self.encode_model.load_model(epochs=self.epochs,
                                         path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/",
                                         name=self.dataset.name)
            self.encode_model.load_word_vectors(epochs=self.epochs,
                                                path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/",
                                                name=self.dataset.name)
        self.encode_model.encode_problems(self.dataset.problem_id_to_index, self.dataset.texts_list)

    def prediction_train(self):
        self.prediction_model.train(self.encode_model, self.dataset.train_set, self.dataset.validation_set)

    def compute_predictions(self):
        self.labels, self.predictions = self.prediction_model.compute_predictions(self.dataset.test_set)
        self.save_predictions()
