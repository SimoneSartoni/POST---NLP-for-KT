import numpy as np


class partition_set:
    def __init__(self, name, path):
        self.name = name
        self.path = path

        # series of sequences of problems
        self.problems = []
        self.labels = []
        self.lengths = []
        self.timestamps = []
        self.skills = []

        self.number_of_users = 0
        self.labels_mean = 0.0
        self.avg_number_of_interactions_per_user = 0
        self.number_interactions = 0
        self.all_1_predictor_precision = 0.0
        self.all_0_predictor_precision = 0.0

    def _compute_interactions_metadata(self):
        self.number_of_users = len(self.problems)
        self.number_interactions = np.sum(self.lengths)
        self.avg_number_of_interactions_per_user = np.mean(self.lengths)
        label_sum = 0
        for label, real_len in list(zip(self.labels, self.lengths)):
            label_sum += np.sum(label[0:real_len])
        self.labels_mean = float(label_sum) / float(np.sum(self.lengths))
        self.all_1_predictor_precision = self.labels_mean
        self.all_0_predictor_precision = 1.0 - self.labels_mean

    def set_interactions(self, problem_ids, labels, lengths, timestamps, skills=None):
        self.problems = problem_ids
        self.labels = labels
        self.lengths = lengths
        self.timestamps = timestamps
        self.skills = skills
        self._compute_interactions_metadata()

    def write_dataset_info(self, f):
        items = dict(vars(self).items())
        f.write("\n")
        del [items['path'], items['problems'], items['labels'], items['lengths'], items['timestamps']]
        f.write(str(items))
