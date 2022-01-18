import ast

import pandas as pd
import numpy as np
from code.garbage.dataset import dataset as Dataset
from code.utils.utils import try_parsing_date


def import_questions_text(datasets_dict):
    datasets = {}
    for el in datasets_dict:
        if el.name == "poj_texts":
            problems = pd.read_csv(el.path, low_memory=False, sep='\n', names=["data"])
        elif el.name == "junyi_texts":
            problems = pd.read_csv(el.path, low_memory=False, sep='#')
        else:
            problems = pd.read_csv(el.path, low_memory=False)
        datasets[el.name] = problems
    return datasets


# Load already processed interactions for assistments_2012 dataset
def import_assistments_2009():
    data = pd.read_csv('C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/data/assistments/2009_2010/non_skill_builder_data_new.csv', error_bad_lines=False)
    real_lens = []
    problems = []
    corrects = []
    timestamps = []
    for user, problem in data.groupby("user_id"):
        correct_answer = problem['correct']
        correct = []
        problem_df = problem["problem_id"]
        problem_list = []
        timestamp_df = problem["order_id"]
        timestamp_list = []
        k = 0
        for p, c, t in list(zip(*(problem_df, correct_answer, timestamp_df))):
            problem_list.append(int(p))
            timestamp_list.append(int(t))
            if c == 1:
                correct.append(1.0)
            else:
                correct.append(0.0)
            k += 1
        if k > 1:
            real_lens.append(k)
            problems.append(problem_list)
            corrects.append(correct)
            timestamps.append(timestamp_list)

    dataset = Dataset(name="assistments_2009", prefix="datasets_with_duplicates/",
                      path='C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/data/assistments/2009_2010/non_skill_builder_data_new.csv')
    dataset.set_interactions(problems, real_lens, corrects, timestamps=timestamps, standard_timestamps=False,
                             validation_percentage=0.2, test_percentage=0.2)
    return dataset


def import_assistments_2012():
    data = pd.read_csv('C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/data/assistments/2012_2013/2012-2013-data-with-predictions-4-final.csv', index_col=False)
    real_lens = []
    problems = []
    corrects = []
    timestamps = []
    for user, problem in data.groupby("user_id"):
        correct_answer = problem['correct']
        correct = []
        problem_df = problem["problem_id"]
        problem_list = []
        timestamp_df = problem["end_time"]
        timestamp_list = []
        y = []
        k = 0
        for p, c, t in list(zip(*(problem_df, correct_answer, timestamp_df))):
            problem_list.append(int(p))
            timestamp_list.append(try_parsing_date(t))
            if c == 1:
                y.append(1.0)
                correct.append(1.0)
            else:
                y.append(-1.0)
                correct.append(0.0)
            k += 1
        if k > 1:
            timestamps.append(timestamp_list)
            real_lens.append(k)
            problems.append(problem_list)
            corrects.append(correct)

    dataset = Dataset(name="assistments_2012",
                      path='C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/data/assistments/2012_2013/2012-2013-data-with-predictions-4-final.csv',
                      prefix="datasets_with_duplicates/")
    dataset.set_interactions(problems, real_lens, corrects, timestamps=timestamps, standard_timestamps=True,
                             validation_percentage=0.1, test_percentage=0.2)
    return dataset


# Load already processed interactions for assistments_2012 dataset npz
def import_assistments_2012_npz():
    data = np.load('C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/data/assistments/2012_2013/2012'
                   '-2013-data-with-predictions-4-final.csv.npz')
    y, problems, real_lens, pro_num, timestamps = data['y'], data['problem'], data['real_len'], \
                                                          data['problem_num'], data['time']

    assistment_dataset = Dataset(name="assistments_2012_npz", path="C:/thesis_2/TransformersForKnowledgeTracing"
                                                                   "/Knowledge_Tracing/data/assistments/2012_2013"
                                                                   "/2012-2013-data-with-predictions-4-final.csv.npz",
                                 prefix="datasets_with_duplicates/")
    assistment_dataset.set_interactions(problems, real_lens, y, timestamps=timestamps, standard_timestamps=True,
                                        validation_percentage=0.2, test_percentage=0.2)
    return assistment_dataset


# Load and process interactions for junyi dataset
def import_junyi_interactions():
    data = pd.read_csv('C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/data/Junyi/junyi.csv', sep='\n', names=['data'])
    data = data['data']
    index = range(0, len(data) // 4)
    real_len_index = [el * 4 for el in index]
    real_lens = [int(data[x]) for x in real_len_index]
    problem_index = [el * 4 + 1 for el in index]
    problem_data = [data[x].split(',') for x in problem_index]
    corrects_index = [el * 4 + 2 for el in index]
    corrects_data = [data[x].split(',') for x in corrects_index]
    timestamps_index = [el * 4 + 3 for el in index]
    timestamps_data = [data[x].split(',') for x in timestamps_index]
    problems = []
    corrects = []
    timestamps = []
    for problem, correct, timestamp, real_len in list(zip(*(problem_data, corrects_data, timestamps_data, real_lens))):
        correct2 = [float(x) for x in correct]
        problem2 = [int(x) for x in problem]
        time = [try_parsing_date(x) for x in timestamp]
        problems.append(problem2)
        corrects.append(correct2)
        timestamps.append(time)
    junyi_dataset = Dataset(name="junyi", path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/data/Junyi/junyi.csv",
                            prefix="datasets_with_duplicates/")
    junyi_dataset.set_interactions(problems, real_lens, corrects, timestamps, standard_timestamps=True,
                                   validation_percentage=0.2, test_percentage=0.2)
    return junyi_dataset


# Load and process interactions of poj dataset
def import_poj_interactions():
    data = pd.read_csv('C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/data/poj/poj_log.csv')
    real_lens = []
    problems = []
    corrects = []
    timestamps = []
    for user, problem in data.groupby("User"):
        correct_answer = problem['Result']
        correct = []
        problem_df = problem["Problem"]
        problem_list = []
        timestamp = problem["Submit Time"]
        timestamp_list = []
        k = 0
        for p, c, t in list(zip(*(problem_df, correct_answer, timestamp))):
            problem_list.append(p)
            timestamp_list.append(try_parsing_date(t))
            if c == "Accepted":
                correct.append(1.0)
            else:
                correct.append(0.0)
            k += 1
        if k > 1:
            real_lens.append(k)
            problems.append(problem_list)
            corrects.append(correct)
            timestamps.append(timestamp_list)
    poj_dataset = Dataset(name="poj", path='C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/data/poj/poj_log.csv',
                          prefix="datasets_with_duplicates/")
    poj_dataset.set_interactions(problems, real_lens, corrects, timestamps=timestamps, standard_timestamps=True,
                                 validation_percentage=0.2, test_percentage=0.2)
    return poj_dataset
