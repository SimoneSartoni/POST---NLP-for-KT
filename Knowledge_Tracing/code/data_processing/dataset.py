import ast
import json
import csv
import os
from datetime import datetime

import numpy as np
import pandas as pd
from mlflow import log_artifact



from Knowledge_Tracing.code.data_processing.partition_set import partition_set
from Knowledge_Tracing.code.utils.utils import write_txt, parse_datetime_list
from sklearn.model_selection import train_test_split
from Knowledge_Tracing.code.visualization.histogram import *
from Knowledge_Tracing.code.utils.utils import try_parsing_date


class dataset:
    def __init__(self, name, path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files/", prefix=''):
        self.name = name
        self.path = path
        self.prefix = prefix

        # data
        self.problems = []
        self.labels = []
        self.lengths = []
        self.timestamps = []
        self.standard_timestamps = True
        # partitions of data
        self.train_set = partition_set("train", path)
        self.validation_set = partition_set("validation", path)
        self.test_set = partition_set("test", path)
        self.random_shuffle = 42

        # list of problems we have at least an interaction
        self.interacted_problem_list = []

        # list of texts
        self.texts_list = []
        self.texts_lengths = []
        # dictionary to obtain from problem_id the corresponding index to access its text in texts_list
        self.problem_id_to_index = {}
        # list of problems we have textual information
        self.problems_with_text_known_list = []

        # number of users we have interactions
        self.number_of_users = 0
        # average value for label
        self.labels_mean = 0.0
        # average number of interactions per user
        self.avg_number_of_interactions_per_user = 0
        # total number of interactions equal to the sum of number of known interactions per user
        self.number_interactions = 0
        # equal to labels_mean, equivalent to accuracy in case we return always 1
        self.all_1_predictor_precision = 0.0
        # equal to 1-labels_mean, equivalent to accuracy in case we return always 0
        self.all_0_predictor_precision = 0.0

        self.number_of_texts = 0
        self.avg_words_per_text = 0

        self.interacted_with_text_problem_set = []
        self.number_of_problems_interacted_with_text = 0

    def _compute_interactions_metadata(self):
        self.number_of_users = len(self.problems)
        self.number_interactions = np.sum(self.lengths)
        self.avg_number_of_interactions_per_user = np.mean(self.lengths)
        label_sum = 0
        for label, length in list(zip(self.labels, self.lengths)):
            label_sum += np.sum(label[0:length])
        self.labels_mean = float(label_sum) / float(np.sum(self.lengths))
        self.all_1_predictor_precision = self.labels_mean
        self.all_0_predictor_precision = 1.0 - self.labels_mean

    def _compute_interacted_problem_list(self):
        interacted_problem_list = {}
        for problem in self.problems:
            for p in problem:
                interacted_problem_list[p] = 1
        self.interacted_problem_list = list(interacted_problem_list.keys())

    def set_interactions(self, user_interactions, user_interactions_lengths, user_interactions_labels, timestamps=None,
                         standard_timestamps=False, validation_percentage=0.0, test_percentage=0.2):
        self.problems, self.labels, self.lengths, self.timestamps = user_interactions, user_interactions_labels, \
                                                                    user_interactions_lengths, timestamps
        self.standard_timestamps = standard_timestamps
        train_timestamps, val_timestamps, test_timestamps = None, None, None
        if not timestamps is None:
            self.chronologically_order_interactions()
            train_timestamps, test_timestamps = train_test_split(self.timestamps, test_size=test_percentage,
                                                                 random_state=self.random_shuffle)
            train_timestamps, val_timestamps = train_test_split(train_timestamps, test_size=test_percentage,
                                                                random_state=self.random_shuffle)
        train_ids, test_ids, train_labels, test_labels, train_lengths, test_lengths = \
            train_test_split(user_interactions, user_interactions_labels, user_interactions_lengths,
                             test_size=test_percentage, random_state=self.random_shuffle)

        train_ids, val_ids, train_labels, val_labels, train_lengths, val_lengths = \
            train_test_split(train_ids, train_labels, train_lengths, test_size=validation_percentage,
                             random_state=self.random_shuffle)
        self.test_set.set_interactions(test_ids, test_labels, test_lengths, test_timestamps)
        self.validation_set.set_interactions(val_ids, val_labels, val_lengths, val_timestamps)
        self.train_set.set_interactions(train_ids, train_labels, train_lengths, train_timestamps)
        self._compute_interactions_metadata()
        self._compute_interacted_problem_list()
        # self.save_interactions()

    def save_interactions(self):
        df = pd.DataFrame(
            data=list(zip(self.problems, self.labels, self.lengths, self.timestamps)),
            columns=["problems", "labels", "lengths", "timestamps"], )
        make_dir("C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files/" + self.prefix + \
               self.name)
        path = "C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files/" + self.prefix + \
               self.name + "/" + "interactions.csv"

        df.to_csv(path, sep=',', quoting=csv.QUOTE_ALL)

    def load_interactions(self, standard_timestamps=True):
        path = self.path + self.prefix +  \
               self.name + "/interactions.csv"
        data = pd.read_csv(filepath_or_buffer=path)
        problems, labels, lengths, timestamps = data["problems"], data["labels"], data["lengths"], data["timestamps"]
        problems = [ast.literal_eval(x) for x in problems]
        labels = [ast.literal_eval(x) for x in labels]
        timestamps = [ast.literal_eval(x) for x in timestamps]

        self.set_interactions(problems, lengths, labels, timestamps=timestamps, standard_timestamps=standard_timestamps,
                              validation_percentage=0.1, test_percentage=0.2)

    def _compute_problems_with_text_known_list(self):
        self.problems_with_text_known_list = self.problem_id_to_index.keys()

    def _compute_texts_metadata(self):
        self.number_of_texts = len(self.texts_list)
        self.avg_words_per_text = np.mean([len(x) for x in self.texts_list])

    def set_texts(self, texts, problem_id_to_index):
        self.texts_list = texts
        self.texts_lengths = [len(x) for x in self.texts_list]
        self.problem_id_to_index = problem_id_to_index
        self._compute_problems_with_text_known_list()
        # self.save_texts()
        self._compute_texts_metadata()

    def compute_intersection_texts_and_interactions(self):
        interacted_with_text_problem_dict = {}
        for p in self.interacted_problem_list:
            if p in self.problems_with_text_known_list:
                interacted_with_text_problem_dict[p] = 1
        self.interacted_with_text_problem_set = list(interacted_with_text_problem_dict.keys())
        self.number_of_problems_interacted_with_text = len(self.interacted_with_text_problem_set)
        # self.draw_graphs()
        # self.write_dataset_info()
        # self.save_interactions()
        # self.log_all()

    def set_performances(self, performances):
        self.performances = performances

    def save_texts(self):

        make_dir("C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files/" + self.prefix +
                 self.name)
        file = "C:/thesis_2/TransformersForKnowledgeTracing" \
               "/Knowledge_Tracing/intermediate_files/" + self.prefix + self.name + "/texts_list.json"
        with open(file, "w") as f:
            json.dump(self.texts_list, f)
        file = "C:/thesis_2/TransformersForKnowledgeTracing" \
               "/Knowledge_Tracing/intermediate_files/" + self.prefix + self.name + "/problem_id_to_index.json"
        with open(file, "w") as f:
            json.dump(self.problem_id_to_index, f)

    def log_all(self):
        file = "C:/thesis_2/TransformersForKnowledgeTracing" \
               "/Knowledge_Tracing/intermediate_files/" + self.prefix +self.name
        log_artifact(file)
        file = "C:/thesis_2/TransformersForKnowledgeTracing" \
               "/Knowledge_Tracing/results/" + self.prefix + self.name
        log_artifact(file)

    def load_saved_texts(self):
        file = self.path + self.prefix + self.name + "/texts_list.json"
        with open(file, "r") as f:
            self.texts_list = json.load(f)

        file = self.path + self.prefix + self.name + "/problem_id_to_index.json"
        with open(file, "r") as f:
            x = json.load(f)
        self.problem_id_to_index = {}
        for el in x.keys():
            self.problem_id_to_index[int(el)] = x[el]
        self._compute_problems_with_text_known_list()
        self._compute_texts_metadata()

    def write_dataset_info(self):
        make_dir("C:/thesis_2/TransformersForKnowledgeTracing" \
               "/Knowledge_Tracing/results/" + self.prefix + self.name)
        file = "C:/thesis_2/TransformersForKnowledgeTracing" \
               "/Knowledge_Tracing/results/" + self.prefix + self.name + "/info_" + self.name
        with open(file, 'w') as f:
            items = dict(vars(self).items())
            del [items['path'], items['problems'], items['labels'], items['lengths'], items['train_set'],
                 items['test_set'], items['validation_set'], items['problem_id_to_index'], items['texts_list'],
                 items['problems_with_text_known_list'], items['interacted_problem_list'], items['texts_lengths'],
                 items['interacted_with_text_problem_set'], items['timestamps']]
            f.write(str(items))
            self.train_set.write_dataset_info(f)
            self.validation_set.write_dataset_info(f)
            self.test_set.write_dataset_info(f)
        file = "C:/thesis_2/TransformersForKnowledgeTracing" \
               "/Knowledge_Tracing/results/" + self.prefix + self.name + "/available_problems_" + self.name
        write_txt(file=file, data=self.interacted_with_text_problem_set)

    def draw_graphs(self):
        make_dir("C:/thesis_2/TransformersForKnowledgeTracing" \
                 "/Knowledge_Tracing/results/" + self.prefix + self.name)
        path = "C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/results/" + self.prefix + self.name + "/"
        histogram2(self.lengths, label_x="number of interactions for user", path=path)
        # histogram_percentage(self.lengths, path=path)
        histogram2(self.texts_lengths, label_x="length of text", path=path)

    def chronologically_order_interactions(self):
        ordered_problems, ordered_labels, ordered_timestamps = [], [], []
        for problem, label, length, timestamp in list(zip(self.problems, self.labels, self.lengths, self.timestamps)):
            order = np.argsort(np.array(timestamp))
            ordered_problems.append(list(np.array(problem)[order]))
            ordered_labels.append(list(np.array(label)[order]))
            ordered_timestamps.append(list(np.array(timestamp)[order]))
        self.problems, self.labels, self.timestamps = ordered_problems, ordered_labels, ordered_timestamps
