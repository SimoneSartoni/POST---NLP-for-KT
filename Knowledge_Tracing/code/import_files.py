import pandas as pd
import numpy as np

def import_questions_text(assistments=True, junyi=True, poj=True):
    datasets = {}
    if assistments:
        assistments_problems = pd.read_csv(
            "../data/Assistments/problem_bodies/ASSISTments2012DataSet-ProblemBodies.csv", low_memory=False)
        datasets["assistments"] = assistments_problems
    if junyi:
        junyi_problems = pd.read_csv("../data/Junyi/junyi_question_text.txt", low_memory=False, sep='#')
        datasets["junyi"] = junyi_problems
    if poj:
        poj_problems = pd.read_csv("../data/poj/poj_question_text.txt", low_memory=False, sep='\n', names=["data"])
        datasets["poj"] = poj_problems
    return datasets


# Load already processed interactions for assistments dataset
def import_assistments_interactions():
    data = np.load('../input/assesments-12-13-precessed-data/2012-2013-data-with-predictions-4-final.csv.npz')
    y, problems, real_lens, pro_num = data['y'], data['problem'], data['real_len'], data['problem_num']

    # Modify corrects value to have: 1.0 if answer is correct, -1.0 if wrong, 0.0 if unknown

    corrects = np.where(y == -1.0, -100.0, y)
    corrects = np.where(corrects == 0.0, -1.0, corrects)
    corrects = np.where(corrects == -100.0, 0.0, corrects)
    return problems, real_lens, corrects


# Load and process interactions for junyi dataset
def import_junyi_interactions():
    data = pd.read_csv('../input/junyi-dataset/junyi.csv', sep='\n', names=['data'])
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
    for problem, correct, real_len in list(zip(*(problem_data, corrects_data, real_lens))):
        correct2 = [float(x) for x in correct]
        problem2 = [int(x) for x in problem]
        problem = problem2
        correct = np.where(correct2 == -1.0, -100.0, correct2)
        correct = np.where(correct == 0.0, -1.0, correct)
        correct = np.where(correct == -100.0, 0.0, correct)
        problems.append(problem)
        corrects.append(correct)
    pro_num = len(problem)
    return problems, real_lens, corrects


# Load and process interactions of POJ dataset
def import_poj_interactions(number_to_index):
    data = pd.read_csv('../input/poj-dataset/poj_log.csv')
    real_lens = []
    problems = []
    corrects = []
    for user, problem in data.groupby("User"):
        correct_answer = problem['Result']
        correct = []
        problem_df = problem["Problem"]
        problem_list = []
        k = 0
        for p, c in list(zip(*(problem_df, correct_answer))):
            if p in number_to_index:
                problem_list.append(number_to_index[p])
                if c == "Accepted":
                    correct.append(1.0)
                else:
                    correct.append(-1.0)
                k += 1
        if k > 1:
            real_lens.append(k)
            problems.append(problem_list)
            corrects.append(correct)
    return problems, real_lens, corrects
