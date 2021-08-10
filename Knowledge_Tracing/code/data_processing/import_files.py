import pandas as pd
import numpy as np
from Knowledge_Tracing.code.data_processing.dataset import dataset as Dataset


def import_questions_text(datasets_dict):
    datasets = {}
    for el in datasets_dict:
        if el.name == "poj_texts":
            problems = pd.read_csv(el.path, low_memory=False, sep='\n', names=["data"])
        elif el.name == "junyi_texts":
            problems = pd.read_csv(el.path, low_memory=False, sep='#')
        else:
            problems = pd.read_csv(el.path, low_memory=False, error_bad_lines=False)
        datasets[el.name] = problems
    return datasets


# Load already processed interactions for assistments dataset
def import_assistments_interactions():
    data = np.load('C:/Users/Simone Sartoni/Simone/Universita/5anno/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/data/Assistments/2012_2013/2012-2013-data-with-predictions-4-final.csv.npz')
    y, problems, real_lens, pro_num = data['y'], data['problem'], data['real_len'], data['problem_num']
    assistment_dataset = Dataset(name="Assistments_2012_2013", path="C:/Users/Simone Sartoni/Simone/Universita/5anno/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/data/Assistments/2012_2013/2012-2013-data-with-predictions-4-final.csv.npz")
    # Modify corrects value to have: 1.0 if answer is correct, -1.0 if wrong, 0.0 if unknown
    corrects = np.where(y == -1.0, -100.0, y)
    corrects = np.where(corrects == 0.0, -1.0, corrects)
    corrects = np.where(corrects == -100.0, 0.0, corrects)
    assistment_dataset.set_interactions(problems, real_lens, corrects, y)
    return assistment_dataset


# Load and process interactions for junyi dataset
def import_junyi_interactions():
    data = pd.read_csv('C:/Users/Simone Sartoni/Simone/Universita/5anno/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/data/Junyi/junyi.csv', sep='\n', names=['data'])
    data = data['data']
    index = range(0, len(data) // 4)
    real_len_index = [el * 4 for el in index]
    real_lens = [int(data[x]) for x in real_len_index]
    problem_index = [el * 4 + 1 for el in index]
    problem_data = [data[x].split(',') for x in problem_index]
    corrects_index = [el * 4 + 2 for el in index]
    corrects_data = [data[x].split(',') for x in corrects_index]
    problems = []
    corrects = []
    target_values = []
    for problem, correct, real_len in list(zip(*(problem_data, corrects_data, real_lens))):
        correct2 = [float(x) for x in correct]
        problem2 = [int(x) for x in problem]
        problem = problem2
        correct = np.where(correct2 == -1.0, -100.0, correct2)
        correct = np.where(correct == 0.0, -1.0, correct)
        correct = np.where(correct == -100.0, 0.0, correct)

        problems.append(problem)
        corrects.append(correct)
        target_values.append(correct2)
    junyi_dataset = Dataset(name="junyi", path="C:/Users/Simone Sartoni/Simone/Universita/5anno/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/data/Junyi/junyi.csv")
    junyi_dataset.set_interactions(problems, real_lens, corrects, target_values)
    pro_num = len(problem)
    return junyi_dataset


# Load and process interactions of POJ dataset
def import_poj_interactions(number_to_index):
    data = pd.read_csv('C:/Users/Simone Sartoni/Simone/Universita/5anno/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/data/poj/poj_log.csv')
    real_lens = []
    problems = []
    corrects = []
    target_values = []
    for user, problem in data.groupby("User"):
        correct_answer = problem['Result']
        correct = []
        problem_df = problem["Problem"]
        problem_list = []
        y = []
        k = 0
        for p, c in list(zip(*(problem_df, correct_answer))):
            if p in number_to_index:
                problem_list.append(number_to_index[p])
                if c == "Accepted":
                    correct.append(1.0)
                    y.append(1.0)
                else:
                    correct.append(-1.0)
                    y.append(0.0)
                k += 1
        if k > 1:
            real_lens.append(k)
            problems.append(problem_list)
            corrects.append(correct)
            target_values.append(y)

    poj_dataset = Dataset(name="poj", path='C:/Users/Simone Sartoni/Simone/Universita/5anno/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/data/poj/poj_log.csv')
    poj_dataset.set_interactions(problems, real_lens, corrects, target_values)
    return poj_dataset
