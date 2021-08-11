import numpy as np
from Knowledge_Tracing.code.utils.utils import write_txt


class dataset:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.target_mean = None
        # series of sequences of problems
        self.problems = None
        self.problems_with_text_known = None
        self.number_of_series = 0
        self.average_number_of_interactions_per_series = 0
        self.corrects = None
        self.real_lens = None
        self.target_values = None
        self.number_interactions = 0
        self.all_1_predictor_precision = 0.0
        self.all_0_predictor_precision = 0.0
        self.texts = None
        self.problem_id_to_index = None
        self.problems_with_text_set = None
        self.problems_interacted_set = None
        self.problems_text_and_interacted_set = None
        self.real_lens_with_text = []
        self.average_number_of_interacted_and_text_per_series = 0
        self.performances = None

    def set_interactions(self, problems, real_lens, corrects, target_values):
        self.problems = problems
        self.corrects = corrects
        self.real_lens = real_lens
        self.number_of_series = len(real_lens)
        self.target_values = target_values
        self.number_interactions = len(real_lens)
        self.average_number_of_interactions_per_series = np.mean(self.real_lens)
        target_sum = 0
        for target_value, real_len in list(zip(target_values, real_lens)):
            target_sum += np.sum(target_value[0:real_len])
        self.target_mean = float(target_sum)/float(np.sum(real_lens))
        self.all_1_predictor_precision = self.target_mean
        self.all_0_predictor_precision = 1.0 - self.target_mean

    def set_texts(self, texts, problem_id_to_index, problems_with_text_set, problems_interacted_set, problems_text_and_interacted_set):
        self.texts = texts
        self.problem_id_to_index = problem_id_to_index
        self.problems_with_text_set = problems_with_text_set
        self.problems_interacted_set = problems_interacted_set
        self.problems_text_and_interacted_set = problems_text_and_interacted_set
        self.problems_with_text_known = []
        for p in self.problems:
            problems_list = []
            for el in p:
                if el in self.problems_text_and_interacted_set:
                    problems_list.append(el)
            self.problems_with_text_known.append(problems_list)
            self.real_lens_with_text.append(len(problems_list))
        self.average_number_of_interacted_and_text_per_series = np.mean([len(p) for p in self.problems_with_text_known])

    def set_performances(self, performances):
        self.performances = performances

    def write_dataset_info(self):
        file = "C:/Users/Simone Sartoni/Simone/Universita/5anno/thesis_2/NLPforKT" \
               "/Knowledge_Tracing/logs/info_" + self.name
        with open(file, 'w') as f:
            items = dict(vars(self).items())
            del [items['path'], items['problems'], items['corrects'], items['real_lens'], items['target_values'],  items['problem_id_to_index'], items['texts'], items['problems_with_text_set'], items['problems_interacted_set'], items['problems_text_and_interacted_set'], items['problems_with_text_known'], items['real_lens_with_text']]
            f.write(str(items))
        file = "C:/Users/Simone Sartoni/Simone/Universita/5anno/thesis_2/NLPforKT" \
               "/Knowledge_Tracing/logs/available_problems_" + self.name
        write_txt(file=file, data=self.problems_text_and_interacted_set)
