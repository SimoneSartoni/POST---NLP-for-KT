import pandas as pd
import os
import nltk
# data_processing hunspell
from nltk.corpus import stopwords

# eng_dict = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')


def escape_values(question):
    text = str(question).replace(' ', '#').replace('/', '#slash#').replace('<', '#lessthan#').replace('>',
                                                                                                              '#morethan#').replace(
        ",", "#comma#").replace(";", "#semicolon#").replace(".", "#point#").replace("?", "#questionmark#").replace(
        "!", "exclamationpoint").replace("=", "#equal#").replace("\\", "#slash#").replace("%",
                                                                                          "#percentage#").replace(
        "\\t", "#").replace("\\n", "#").replace("\t", "#").replace("\n", "#").replace('\"',
                                                                                      "#quotationmark#").replace(
        "(", "#openroundbracket#").replace(")", "#closeroundbracket#").replace("[", "#opensquarebracket#").replace(
        "]", "#closesquarebracket#").replace("_", "#underscore#").replace("&", "#ampersand#").replace("}",
                                                                                                      "#closebrace#").replace(
        "{", "#openbrace#").replace("+", "#plus#").replace("-", "#minus#").replace("*", "#multiplication#").replace(
        "€", "#euros#").replace("$", "#dollar#").replace("^", "#powerof#exponent#")
    return str(text).split('#')


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
    for body in bodies:
        text = escape_values(body)
        #text = remove_words_not_in_english_dict(text)
        text = remove_stopwords(text)
        text = remove_issues(text)
        texts.append(text)
    return problem_ids, texts


def junyi_process_questions(df):
    problem_names, questions, question_descriptions = df['question_name'], df['chinese_question'], df[
        'chinese_question_desc']
    texts = []
    nltk.download('stopwords')
    problem_ids = range(0, len(questions))
    for index in range(0, len(questions)):
        text = escape_values(questions[index])
        text_desc = escape_values(question_descriptions[index])
        text = list(set(text) | set(text_desc))
        #text = remove_words_not_in_english_dict(text)
        text = remove_stopwords(text)
        text = remove_issues(text)
        texts.append(text)
    return texts, problem_ids


def generate_questions_poj(df):
    questions = []
    number_to_index = dict({})
    index = 0
    for row in df['data']:
        if '#' in row:
            array = row.split('#')
            if array[0].isdigit():
                questions.append([])
                index = index + 1
                number = int(array[0])
                number_to_index[number] = index
                questions[index] = array[1]

        else:
            new = str(questions[index]) + str(row)
            questions[index] = new
    questions.append([])
    problem_ids = number_to_index.keys()
    return questions, problem_ids, number_to_index


def poj_process_bodies(df):
    questions, number_to_index = generate_questions_poj(df)
    texts = []
    nltk.download('stopwords')
    for index in range(0, len(questions)):
        text = escape_values(questions[index])
        # text = remove_words_not_in_english_dict(text)
        text = remove_stopwords(text)
        text = remove_issues(text)
        texts.append(text)
    return texts, number_to_index


def generate_text_and_interacted_sets(problem_ids, problems, real_lens, corrects):
    problems_interacted_set = set([])
    for problem, correct, real_len in list(zip(*(problems, corrects, real_lens))):
        # TF-IDF:
        problems_set_with_results = problems_interacted_set.union(set(problem))
    problems_with_text_set = set(problem_ids)
    print(problems_with_text_set)
    print(problems_interacted_set)
    problems_text_and_interacted_set = problems_with_text_set.intersection(problems_set_with_results)
    print(len(problems_text_and_interacted_set))
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
        new_problem, new_correct, new_real_len = new_problem[0:-2], new_correct[0:-2], new_real_len-1
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

