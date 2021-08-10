from time import time


class predictor:
    def __init__(self):
        self.time_to_predict = None
        self.predictions = dict({})
        self.labels = []

    def compute_predictions(self, dataset, models=[]):
        t = time()
        for model in models:
            self.predictions[model.name] = []
        self.labels = []
        for problem, correct, real_len in list(zip(*(dataset.problems, dataset.corrects, dataset.real_lens))):
            target_problem = problem[real_len-1]
            input_problems = problem[:real_len-1]
            input_ids = []
            correct_ids = []
            for p, c in list(zip(input_problems, correct[:real_len-1])):
                if p in dataset.problem_id_to_index:
                    input_ids.append(dataset.problem_id_to_index[p])
                    correct_ids.append(c)
            for model in models:
                if target_problem in dataset.problem_id_to_index:
                    self.predictions[model.name].append(model.compute_problem_score(problems=input_ids, corrects=correct_ids, target_problem=dataset.problem_id_to_index[target_problem]))
                else:
                    self.predictions[model.name].append(round(dataset.all_1_predictor_precision))
            self.labels.append(correct[real_len-1])
        self.time_to_predict = round((time() - t) / 60, 2)
        return self.labels, self.predictions
