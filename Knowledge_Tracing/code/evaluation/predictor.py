import time


class predictor:
    def __init__(self, name):
        self.name = name
        self.time_to_predict = None
        self.predictions = dict({})

    def compute_predictions(self, problems, corrects, real_lens, target_problems, models=[]):
        t = time()
        for model in models:
            self.predictions[model.name] = []
        for problem, correct, real_len, target_problem in list(zip(*(problems, corrects, real_lens, target_problems))):
            for model in models:
                self.predictions[model.name].append(model._compute_problem_score(problems=problem, corrects=corrects, target_problem=target_problem))
        self.time_to_predict = round((time() - t) / 60, 2)
        return self.predictions
