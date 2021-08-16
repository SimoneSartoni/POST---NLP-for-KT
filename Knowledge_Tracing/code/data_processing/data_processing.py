import gc
import re
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import os
import nltk
# data_processing hunspell
import torch
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import psutil
from Knowledge_Tracing.code.data_processing.dataset_pytorch import dataset


# eng_dict = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')

def escape_values(question):
    def replace(text):
        text = str(text).replace(' ', '#').replace('/', '#slash#').replace('<', '#lessthan#').replace('>','#morethan#').replace(",", "#").replace(";", "#").replace(".", "#").replace("?", "#").replace(
            "!", "exclamationpoint").replace("=", "#equal#").replace("\\", "#").replace("%", "#percentage#").replace(
            "\\t", "#").replace("\\n", "#").replace("\t", "#").replace("\n", "#").replace('\"', "##").replace(
            "(", "#openroundbracket#").replace(")", "#closeroundbracket#").replace("[", "#opensquarebracket#").replace(
            "]", "#closesquarebracket#").replace("_", "#underscore#").replace("&", "#ampersand#").replace("}","#closebrace#").replace(
            "{", "#openbrace#").replace("+", "#plus#").replace("-", "#minus#").replace("*", "#multiplication#").replace(
            "€", "#euros#").replace("$", "#dollar#").replace("^", "#powerof#exponent#")
        words = str(text).split('#')
        words = list(set(words))
        return words

    texts = ""
    question = str(question).lower()
    # names = ['head', 'p', 'sup', 'br', 'b', 'td', 'tr', 'table', 'i', 'em', 'sub', 'tbody', 'strong', 'span', 'li', 'ul', ]
    soup = BeautifulSoup(question, "html.parser")
    for el in soup.find_all('img'):
        texts = texts + ' ' + str(el.unwrap())
    p = re.compile(r'<.*?>')
    texts = texts + ' ' + p.sub('', question)
    return replace(texts)


"""def remove_words_not_in_english_dict(text):
    set1 = {}
    for i in range(0, len(text)):
        text[i].lower()
        if eng_dict.spell(text[i]):
            set1.add(text[i])
    return list(set1)"""


def remove_stopwords(text):
    for i in stopwords.words('english'):
        i = i.lower()
        if text.count(i) > 0:
            text.remove(i)
    return text


def remove_issues(text):
    if text.count('Timeout') > 0:
        text.remove('Timeout')
    if text.count('TIMEOUT') > 0:
        text.remove('TIMEOUT')
    if text.count('ISSUE') > 0:
        text.remove('ISSUE')
    if text.count('underscore') > 0:
        text.remove('underscore')
    return text



def assistments_process_bodies(df):
    problem_ids, assistment_ids, bodies = df['problem_id'], df['assistment_id'], df['body']
    texts = []
    nltk.download('stopwords')
    problem_id_to_index = {}
    index = 0
    for body, problem_id in list(zip(bodies, problem_ids)):
        text = escape_values(body)
        # text = remove_words_not_in_english_dict(text)
        text = remove_stopwords(text)
        text = remove_issues(text)
        texts.append(text)
        problem_id_to_index[problem_id] = index
        index += 1
    return texts, problem_id_to_index


def junyi_process_questions(df):
    problem_names, questions, question_descriptions = df['question_name'], df['chinese_question'], df[
        'chinese_question_desc']
    texts = []
    problem_id_to_index = {}
    # nltk.download('stopwords')
    index = 0
    for question_id in range(0, len(questions)):
        text = escape_values(questions[question_id])
        text_desc = escape_values(question_descriptions[question_id])
        text = list(set(text) | set(text_desc))
        # text = remove_words_not_in_english_dict(text)
        text = remove_stopwords(text)
        text = remove_issues(text)
        if len(text) > 0:
            texts.append(text)
            problem_id_to_index[question_id] = index
        index += 1
    return texts, problem_id_to_index


def generate_questions_poj(df):
    questions = []
    problem_id_to_index = {}
    index = -1
    for row in df["data"]:
        if '#' in row:
            array = row.split('#')
            if array[0].isdigit():
                questions.append(" ")
                index += 1
                id = int(array[0])
                problem_id_to_index[id] = index
                questions[index] = array[1]

        else:
            new = str(questions[index]) + str(row)
            questions[index] = new
    questions.append([])
    return questions, problem_id_to_index


def poj_process_bodies(df):
    questions, problem_id_to_index = generate_questions_poj(df)
    # nltk.download('stopwords')
    copy = problem_id_to_index.keys()
    for id in copy:
        text = escape_values(questions[problem_id_to_index[id]])
        # text = remove_words_not_in
        text = remove_issues(text)
        questions[problem_id_to_index[id]] = text
    return questions, problem_id_to_index


def generate_text_and_interacted_sets(problem_ids, problems):
    problems_text_and_interacted_set = []
    problems_with_text_set = problem_ids
    problems_interacted_set = []
    for problem in problems:
        for p in problem:
            if p not in problems_interacted_set:
                problems_interacted_set.append(p)
                if p in problems_with_text_set:
                    problems_text_and_interacted_set.append(p)
    return problems_with_text_set, problems_interacted_set, problems_text_and_interacted_set


def remove_problems_without_text_or_interactions(problems_list, correctness_list, problems_set_text_and_results):
    new_problems_list = []
    new_correctness_list = []
    for p in range(0, len(problems_list)):
        if p in problems_set_text_and_results:
            new_problems_list.append(problems_list[p])
            new_correctness_list.append(correctness_list[p])
    new_real_len = len(new_problems_list)
    return new_problems_list, new_correctness_list, new_real_len


def generate_data_for_predictions(problems_set_text_and_results, problems, corrects, real_lens):
    labels = []
    new_problems = []
    new_corrects = []
    new_real_lens = []
    target_problems = []
    for problem, correct, real_len in list(zip(*(problems, corrects, real_lens))):
        # TF-IDF:
        target_problem = problem[real_len - 1]
        target_correct = correct[real_len - 1]
        new_problem, new_correct, new_real_len = remove_problems_without_text_or_interactions
        new_problem, new_correct, new_real_len = new_problem[0:-2], new_correct[0:-2], new_real_len - 1
        if target_problem in problems_set_text_and_results:
            new_problems.append(new_problem)
            new_corrects.append(new_correct)
            new_real_lens.append(new_real_len)
            target_problems.append(target_problem)
            if target_correct == 1.0:
                labels.append(1)
            else:
                labels.append(0)
    return new_problems, new_corrects, new_real_lens, target_problems, labels


def train_test_split(data, labels, split=0.8):
    n_samples = len(data)
    # x is your dataset
    training_data, test_data = data[:int(n_samples*split)], data[int(n_samples*split):]
    training_labels, test_labels = labels[:int(n_samples*split)], labels[int(n_samples*split):]
    return training_data, test_data, training_labels, test_labels


def generate_sequences_for_training_RKT(problems, real_lens, corrects, batch_size=64):
    """Extract sequences from dataframe.
    Arguments:
        df (pandas Dataframe): output by prepare_data.py
        max_length (int): maximum length of a sequence chunk
        train_split (float): proportion of data to use for training
    """

    params = {'batch_size': batch_size,
              'shuffle': True}
    process = psutil.Process(os.getpid())
    gc.enable()
    # data = np.load('../input/assesments-12-13-precessed-data/2012-2013-data-with-predictions-4-final.csv.npz')
    skill_num, pro_num = 0, len(problems)
    timestamps = []
    item_ids = [torch.tensor(i).type(torch.cuda.LongTensor) for i in problems]
    timestamp = [
        torch.tensor([(t - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's') for t in timestamp]).type(
            torch.cuda.LongTensor) for timestamp in timestamps]
    labels = [torch.tensor(i).type(torch.cuda.LongTensor) for i in y]
    item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long).cuda(), i))[:-1] for i in item_ids]
    # skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s))[:-1] for s in skill_ids]
    label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long).cuda(), l))[:-1] for l in labels]

    batches = list(zip(item_inputs, label_inputs, item_ids, timestamp, labels))
    seq_lists = list(zip(*batches))
    inputs_and_ids = [pad_sequence(seqs, batch_first=True, padding_value=0)
                      for seqs in seq_lists[0:4]]
    labels = pad_sequence(seq_lists[-1], batch_first=True, padding_value=-1)  # Pad labels with -1
    train_data, test_data, training_labels, test_labels = train_test_split(data=list(zip(*inputs_and_ids)), labels=labels, split=0.8)

    # TODO
    print("pro_pro_dense computation")
    pro_pro_dense = []
    training_set = dataset(train_data, training_labels)
    # training_generator = torch.utils.data.DataLoader(training_set, **params)
    test_set = dataset(test_data, test_labels)
    # test_generator = torch.utils.data.DataLoader(test_set, **params)
    # validation_set = dataset(val_data, val_labels)
    # validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    return training_set, test_set, pro_pro_dense, pro_num, timestamps
