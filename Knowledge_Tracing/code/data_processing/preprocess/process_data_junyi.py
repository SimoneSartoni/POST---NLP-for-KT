import gc
from datetime import datetime

import numpy as np
import pandas as pd
import itertools
from Knowledge_Tracing.code.utils.utils import try_parsing_date
from Knowledge_Tracing.code.data_processing.preprocess.load_process_junyi_texts import load_process_junyi_texts


def process_data_junyi(min_questions=2, max_questions=50, interactions_filepath="../input/assistmentds"
                                                                                           "-2012/2012-2013-data-with"
                                                                                           "-predictions-4-final.csv",
                                  texts_filepath='../input/', output_filepath="/kaggle/working/", n_rows=None,
                                  n_texts=None, personal_cleaning=True, make_sentences_flag=True):
    data = pd.read_csv(interactions_filepath, sep='\n', names=['data'], nrows =n_rows)
    print("data read")
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
    user_id = 0
    del data
    gc.collect()
    problems = list(itertools.chain(problem_data))
    del problem_data
    gc.collect()
    corrects = list(itertools.chain(corrects_data))
    del corrects_data
    gc.collect()
    timestamps = list(itertools.chain(timestamps_data))
    del timestamps_data
    gc.collect()
    index = 0
    users = []
    for real_len in real_lens:
        for i in range(0, real_len):
            users.append(index)
        index += 1
    del real_lens
    print(len(users))
    print(len(problems))
    gc.collect()
    dictionary = {'user_id': users, 'problem_id': problems, 'correct': corrects, "timestamp": timestamps}
    train_df = pd.DataFrame(dictionary)

    print("shape of dataframe :", train_df.shape)

    # Step 4 - Sort interactions according to timestamp
    train_df = train_df.sort_values(["timestamp"], ascending=True)

    train_df = train_df.drop_duplicates(subset=['user_id', 'problem_id'], keep='first').reset_index(drop=True)

    # Step 1 - Remove users with less than a certain number of answers
    # train_df = train_df.groupby('user_id').tail(max_questions)
    print("shape after at least 2 interactions:", train_df.shape)

    # Step 1 - Remove users with less than a certain number of answers
    train_df = train_df.groupby('user_id').filter(lambda q: len(q) >= min_questions).copy()
    print("shape after at least " + min_questions + " interactions:", train_df.shape)

    # Step 5 - Compute number of unique skills ids and number of unique question ids
    questions_ids = train_df['problem_id'].unique()
    n_ids = len(questions_ids)
    print("no. of problems :", n_ids)
    print("shape after exclusion:", train_df.shape)
    print("Get texts, intersection...")

    # Step 6 - Remove questions interactions we do not have text
    texts_df = load_process_junyi_texts(personal_cleaning=personal_cleaning, texts_filepath=texts_filepath, n_texts=n_texts, make_sentences_flag=make_sentences_flag)
    train_df = train_df.loc[train_df['problem_id'].isin(texts_df['problem_id'])]
    texts_df = texts_df.loc[texts_df['problem_id'].isin(train_df['problem_id'])]

    n_ids = len(questions_ids)
    print("no. of problems :", n_ids)
    print("shape after exclusion:", train_df.shape)
    texts_df['question_id'], _ = pd.factorize(texts_df['problem_id'], sort=True)
    train_df['question_id'], _ = pd.factorize(train_df['problem_id'], sort=True)

    texts_df.to_csv(output_filepath+'texts_processed.csv')
    train_df.to_csv(output_filepath+'interactions_processed.csv')
    return train_df, texts_df
