import numpy as np
from Knowledge_Tracing.code.utils.utils import write_txt


class dataset:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.target_mean = None
        self.problems = None
        self.number_of_series = 0
        self.average_number_of_interactions_per_series = 0
        self.corrects = None
        self.real_lens = None
        self.target_values = None
        self.number_interactions = 0
        self.all_1_predictor_precision = 0.0
        self.all_0_predictor_precision = 0.0
        self.texts = None
        self.problem_ids = None
        self.problems_with_text_set = None
        self.problems_interacted_set = None
        self.problems_text_and_interacted_set = None

    def set_interactions(self, problems, real_lens, corrects, target_values):
        self.problems = problems
        self.corrects = corrects
        self.real_lens = real_lens
        self.target_values = target_values
        self.number_interactions = len(real_lens)
        self.average_number_of_interactions_per_series = np.mean(self.real_lens)
        target_sum = 0
        for target_value, real_len in list(zip(target_values, real_lens)):
            target_sum += np.sum(target_value[0:real_len])
        self.target_mean = float(target_sum)/float(np.sum(real_lens))
        self.all_1_predictor_precision = self.target_mean
        self.all_0_predictor_precision = 1.0 - self.target_mean

    def set_texts(self, texts, problem_ids, problems_with_text_set, problems_interacted_set, problems_text_and_interacted_set):
        self.texts = texts
        self.problem_ids = problem_ids
        self.problems_with_text_set = problems_with_text_set
        self.problems_interacted_set = problems_interacted_set
        self.problems_text_and_interacted_set = problems_text_and_interacted_set

    def write_dataset_info(self):
        file = "C:/Users/Simone Sartoni/Simone/Universita/5anno/thesis_2/TransformersForKnowledgeTracing" \
               "/Knowledge_Tracing/logs/info_" + self.name 0
        with open(file, 'w') as f:
            items = dict(vars(self).items())
            del [items['path'], items['problems'], items['corrects'], items['real_lens'], items['target_values'],  items['problem_ids'], items['texts'], items['problems_with_text_set'], items['problems_interacted_set'], items['problems_text_and_interacted_set']]
            f.write(str(items))
