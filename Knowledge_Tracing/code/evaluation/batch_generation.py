import numpy as np


def generate_similarities(encoding_model, problems, corrects, lengths):
    input_similarities = []
    labels = []
    for problem, correct, length in list(zip(problems, corrects, lengths)):
        input_ids = problem[0:length - 1]
        input_corrects = [-1.0 if x == 0.0 else 1.0 for x in correct[0:length - 1]]
        target_problem = problem[length - 1]
        input_similarities.append(
            encoding_model.compute_similarities(input_problems=input_ids, corrects=input_corrects,
                                                 target_problem=target_problem))
        labels.append(correct[length - 1])
    input_similarities = np.array(input_similarities)
    return input_similarities, labels


def generate_similarity_scores(encoding_model, problems, corrects, lengths):
    input_similarities = []
    labels = []
    for problem, correct, length in list(zip(problems, corrects, lengths)):
        input_ids = problem[0:length - 1]
        input_corrects = [-1.0 if x == 0.0 else 1.0 for x in correct[0:length - 1]]
        target_problem = problem[length - 1]
        input_similarities.append(
            encoding_model.compute_problem_score(input_problems=input_ids, corrects=input_corrects,
                                                 target_problem=target_problem))
        labels.append(correct[length - 1])
    input_similarities = np.array(input_similarities).reshape(-1, 1)
    return input_similarities, labels


def generate_features_encoding(encoding_model, problems, corrects, lengths):
    labels = []
    features_encodings = []
    for problem, correct, length in list(zip(problems, corrects, lengths)):
        input_ids = problem[0:length - 1]
        input_corrects = [-1.0 if x == 0.0 else 1.0 for x in correct[0:length - 1]]
        target_problem = problem[length - 1]
        features_encodings.append(
            encoding_model.compute_encoding(input_problems=input_ids, corrects=input_corrects,
                                            target_problem=target_problem))
        labels.append(correct[length - 1])
    features_encodings = np.array(features_encodings)
    return features_encodings, labels