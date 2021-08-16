import json

import numpy as np
from Knowledge_Tracing.code.utils.utils import write_txt


class dataset:
    def __init__(self, name, path):
        self.name = name
        self.path = path

        # series of sequences of problems
        self.users_interactions_problems = []
        self.user_interactions_labels = []
        self.user_interactions_lengths = []
        self.user_interactions_target_values = []

        self.interacted_problem_list = []

        self.user_interactions_with_known_text = []
        self.user_labels_with_known_text = []
        self.users_interactions_with_known_text_lengths = []

        self.texts_list = []
        self.problem_id_to_index = {}
        self.problems_with_text_known_list = []

        self.number_of_users = 0
        self.labels_mean = 0.0
        self.avg_number_of_interactions_per_user = 0
        self.number_interactions = 0
        self.all_1_predictor_precision = 0.0
        self.all_0_predictor_precision = 0.0
        self.avg_number_of_interactions_with_known_text_per_user = 0

        self.number_of_texts = 0
        self.avg_words_per_text = 0

        self.interacted_with_text_problem_set = []

        self.performances = None

    def _compute_interactions_metadata(self):
        self.number_of_users = len(self.user_interactions_lengths)
        self.number_interactions = np.sum(self.user_interactions_lengths)
        self.avg_number_of_interactions_per_user = np.mean(self.user_interactions_lengths)
        target_sum = 0
        for target_value, real_len in list(zip(self.user_interactions_target_values, self.user_interactions_lengths)):
            target_sum += np.sum(target_value[0:real_len])
        self.labels_mean = float(target_sum) / float(np.sum(self.user_interactions_lengths))
        self.all_1_predictor_precision = self.labels_mean
        self.all_0_predictor_precision = 1.0 - self.labels_mean

    def _compute_interacted_problem_list(self):
        interacted_problem_list = []
        for problem in self.users_interactions_problems:
            for p in problem:
                if p not in interacted_problem_list:
                    interacted_problem_list.append(p)
        self.interacted_problem_list = interacted_problem_list

    def set_interactions(self, users_interactions, user_interactions_lengths, user_interactions_labels,
                         user_interactions_target_values):
        self.users_interactions_problems = users_interactions
        self.user_interactions_labels = user_interactions_labels
        self.user_interactions_lengths = user_interactions_lengths
        self.user_interactions_target_values = user_interactions_target_values
        self._compute_interactions_metadata()
        self._compute_interacted_problem_list()

    def _compute_problems_with_text_known_list(self):
        self.problems_with_text_known_list = self.problem_id_to_index.keys()

    def _compute_texts_metadata(self):
        self.number_of_texts = len(self.texts_list)
        self.avg_words_per_text = np.mean([len(x) for x in self.texts_list])

    def set_texts(self, texts, problem_id_to_index):
        self.texts_list = texts
        self.problem_id_to_index = problem_id_to_index
        self._compute_problems_with_text_known_list()
        self.save_texts()
        self._compute_texts_metadata()

    def compute_intersection_texts_and_interactions(self):
        for p in self.users_interactions_problems:
            problems_list = []
            for el in p:
                if el in self.problems_with_text_known_list:
                    problems_list.append(el)
            self.user_interactions_with_known_text.append(problems_list)
            self.users_interactions_with_known_text_lengths.append(len(problems_list))
        self.interacted_with_text_problem_set = set(self.interacted_problem_list).intersection(set(self.problems_with_text_known_list))
        self.avg_number_of_interactions_with_known_text_per_user = np.mean(
            [len(p) for p in self.user_interactions_with_known_text])

    def set_performances(self, performances):
        self.performances = performances

    def save_texts(self):
        file = "C:/thesis_2/TransformersForKnowledgeTracing" \
               "/Knowledge_Tracing/logs/" + self.name + "/texts_list.json"
        a_file = open(file, "w")
        json.dump(self.texts_list, a_file)
        a_file.close()

        file = "C:/thesis_2/TransformersForKnowledgeTracing" \
               "/Knowledge_Tracing/logs/" + self.name + "/problem_id_to_index.json"
        a_file = open(file, "w")
        json.dump(self.problem_id_to_index, a_file)
        a_file.close()

    def load_saved_texts(self, path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/results/"):
        file = path + self.name + "/texts_list.json"
        with open(file, "r") as f:
            self.texts_list = json.load(f)

        file = path + self.name + "/problem_id_to_index.json"
        with open(file, "r") as f:
            self.problem_id_to_index = json.load(f)
        self._compute_problems_with_text_known_list()
        self._compute_texts_metadata()

    def write_dataset_info(self):
        file = "C:/thesis_2/TransformersForKnowledgeTracing" \
               "/Knowledge_Tracing/logs/info_" + self.name
        with open(file, 'w') as f:
            items = dict(vars(self).items())
            del [items['path'], items['users_interactions_problems'], items['user_interactions_labels'],
                 items['user_interactions_lengths'], items['user_interactions_target_values'],
                 items['user_interactions_with_known_text'], items['problem_id_to_index'], items['texts_list'],
                 items['problems_with_text_known_list'], items['interacted_problem_list'],
                 items['interacted_with_text_problem_set'], items['user_labels_with_known_text'],
                 items['users_interactions_with_known_text_lengths']]
            f.write(str(items))
        file = "C:/thesis_2/TransformersForKnowledgeTracing" \
               "/Knowledge_Tracing/logs/available_problems_" + self.name
        write_txt(file=file, data=self.interacted_with_text_problem_set)
