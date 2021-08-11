from time import time


class predictor:
    def __init__(self):
        self.time_to_predict = None
        self.predictions = {}
        self.labels = []

    def compute_predictions(self, dataset, models=[]):
        t = time()
        for model in models:
            self.predictions[model.name] = []
        self.labels = []
        for problem, correct, real_len in list(zip(*(dataset.problems, dataset.corrects, dataset.real_lens))):
            target_problem = problem[real_len-1]
            input_problems = problem[:real_len-1]
            input_corrects = correct[:real_len-1]
            for model in models:
                self.predictions[model.name].append(model.compute_problem_score(input_problems=input_problems, corrects=input_corrects, target_problem=target_problem, default_value=round(dataset.all_1_predictor_precision)))
            if correct[real_len-1] == -1.0:
                self.labels.append(0.0)
            else:
                self.labels.append(1.0)
        self.time_to_predict = round((time() - t) / 60, 2)
        return self.labels, self.predictions
