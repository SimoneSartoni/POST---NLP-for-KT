import json

import numpy as np
import pandas as pd

from Knowledge_Tracing.code.data_processing.partition_set import partition_set
from Knowledge_Tracing.code.utils.utils import write_txt
from sklearn.model_selection import train_test_split
from Knowledge_Tracing.code.visualization.histogram import *


class dataset:
    def __init__(self, name, path):
        self.name = name
        self.path = path

        # data
        self.problems = []
        self.labels = []
        self.lengths = []
        # partitions of data
        self.train_set = partition_set("train", path)
        self.validation_set = partition_set("validation", path)
        self.test_set = partition_set("test", path)

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
        interacted_problem_list = []
        for problem in self.problems:
            for p in problem:
                if not (p in interacted_problem_list):
                    interacted_problem_list.append(p)
        self.interacted_problem_list = interacted_problem_list

    def set_interactions(self, user_interactions, user_interactions_lengths, user_interactions_labels,
                         validation_percentage=0.2, test_percentage=0.2):
        random_state = 42
        self.problems, self.labels, self.lengths = user_interactions, user_interactions_labels, user_interactions_lengths
        train_ids, test_ids, train_labels, test_labels, train_lengths, test_lengths = train_test_split(user_interactions, user_interactions_labels, user_interactions_lengths, test_size=test_percentage, random_state=random_state)
        self.test_set.set_interactions(test_ids, test_labels, test_lengths)
        train_ids, val_ids, train_labels, val_labels, train_lengths, val_lengths = train_test_split(train_ids, train_labels, train_lengths, test_size=validation_percentage, random_state=0)
        self.validation_set.set_interactions(val_ids, val_labels, val_lengths)
        self.train_set.set_interactions(train_ids, train_labels, train_lengths)
        self._compute_interactions_metadata()
        self._compute_interacted_problem_list()
        self.save_interactions()

    def save_interactions(self):
        df = pd.DataFrame(
            data=list(zip(self.problems, self.labels, self.lengths)),
            columns=['problems', 'labels', "lengths"])
        df.to_csv(
            "C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/" + self.name + "/" + "interactions.csv")

    def load_interactions(self):
        path = "C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files/"+self.name+"/interactions.csv"
        data = pd.read_csv(path)
        problems, labels, lengths = data["problems"], data["labels"], data["lengths"]
        self.set_interactions(problems, lengths, labels, 0.2, 0.2)

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
        self.save_texts()
        self._compute_texts_metadata()

    def compute_intersection_texts_and_interactions(self):
        self.interacted_with_text_problem_set = set(self.interacted_problem_list).intersection(
            set(self.problems_with_text_known_list))
        self.number_of_problems_interacted_with_text = len(self.interacted_with_text_problem_set)
        self.write_dataset_info()
        self.draw_graphs()

    def set_performances(self, performances):
        self.performances = performances

    def save_texts(self):
        file = "C:/thesis_2/TransformersForKnowledgeTracing" \
               "/Knowledge_Tracing/logs/" + self.name + "/texts_list.json"
        with open(file, "w") as f:
            json.dump(self.texts_list, f)
        file = "C:/thesis_2/TransformersForKnowledgeTracing" \
               "/Knowledge_Tracing/logs/" + self.name + "/problem_id_to_index.json"
        with open(file, "w") as f:
            json.dump(self.problem_id_to_index, f)

    def load_saved_texts(self):
        path = "C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files/"
        file = path + self.name + "/texts_list.json"
        with open(file, "r") as f:
            self.texts_list = json.load(f)

        file = path + self.name + "/problem_id_to_index.json"
        with open(file, "r") as f:
            x = json.load(f)
        self.problem_id_to_index = {}
        for el in x.keys():
            self.problem_id_to_index[int(el)] = x[el]
        self._compute_problems_with_text_known_list()
        self._compute_texts_metadata()

    def write_dataset_info(self):
        file = "C:/thesis_2/TransformersForKnowledgeTracing" \
               "/Knowledge_Tracing/logs/"+ self.name + "/info_" + self.name
        with open(file, 'w') as f:
            items = dict(vars(self).items())
            del [items['path'], items['problems'], items['labels'], items['lengths'], items['train_set'],
                 items['test_set'], items['validation_set'], items['problem_id_to_index'], items['texts_list'],
                 items['problems_with_text_known_list'], items['interacted_problem_list'], items['texts_lengths'],
                 items['interacted_with_text_problem_set']]
            f.write(str(items))
            self.train_set.write_dataset_info(f)
            self.validation_set.write_dataset_info(f)
            self.test_set.write_dataset_info(f)
        file = "C:/thesis_2/TransformersForKnowledgeTracing" \
               "/Knowledge_Tracing/logs/" + self.name + "/available_problems_" + self.name
        write_txt(file=file, data=self.interacted_with_text_problem_set)

    def draw_graphs(self):
        path = "C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/" + self.name + "/"
        histogram2(self.lengths, label_x="number of interactions for user", path=path)
        # histogram_percentage(self.lengths, path=path)
        histogram2(self.texts_lengths, label_x="lenght of text", path=path)
