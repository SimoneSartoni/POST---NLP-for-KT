import csv
from datetime import datetime

import numpy as np
import pandas as pd

from Knowledge_Tracing.code.utils.utils import try_parsing_date
from Knowledge_Tracing.code.data_processing.get_cloudacademy_texts import get_cloudacademy_texts


def process_data_cloudacademy(min_questions=2, max_questions=50, interactions_filepath="../input/assistmentds"
                                                                                           "-2012/2012-2013-data-with"
                                                                                           "-predictions-4-final.csv",
                                  texts_filepath='../input/', output_filepath="/kaggle/working/", n_rows=None,
                                  n_texts=None, personal_cleaning=True, make_sentences_flag=True):
    input_columns = [
        '_actor_id', 'session_mode', '_time_stamp', 'session_step', 'timer',
        'elapsed_time', 'action', 'correct', 'question_id', 'session_id', 'source',
        '_platform', 'certification_id'
    ]

    print("loading csv.....")
    if n_rows:
        train_df = pd.read_csv(interactions_filepath, names=input_columns, nrows=n_rows)
    else:
        train_df = pd.read_csv(interactions_filepath, names=input_columns)
    print("shape of dataframe :", train_df.shape)
    renaming_dict = {"_actor_id": "user_id", "_time_stamp": "timestamp", "question_id": "problem_id", }
    train_df.rename(renaming_dict)
    # Step 3.1 - Define start, end and elapsed time, fill no timed elapsed time and cap values under a max

    # Step 4 - Sort interactions according to timestamp
    train_df = train_df.sort_values(["timestamp"], ascending=True)

    train_df = train_df.drop_duplicates(subset=['user_id', 'problem_id'], keep='first').reset_index(drop=True)

    # Step 1 - Remove users with less than a certain number of answers
    # train_df = train_df.groupby('user_id').tail(max_questions)
    print("shape after drop duplicated interactions:", train_df.shape)

    # Step 1 - Remove users with less than a certain number of answers
    train_df = train_df.groupby('user_id').filter(lambda q: len(q) >= min_questions).copy()
    print("shape after at least "+str(min_questions)+" interactions:", train_df.shape)

    # Step 2.1 - Fill no skilled question with "no_skill" token
    train_df.fillna("unknown", inplace=True)
    print("shape after filling unknown values:", train_df.shape)

    """# Step 2.2 - Enumerate skill ids and question ids
    train_df['skill'], _ = pd.factorize(train_df['skill'], sort=True)
    print("shape after factorize:", train_df.shape)"""

    # Step 5 - Compute number of unique skills ids and number of unique question ids
    questions_ids = train_df['problem_id'].unique()
    n_ids = len(questions_ids)
    # n_skills = len(train_df['skill'].unique())
    print("no. of problems :", n_ids)
    # print("no. of skills: ", n_skills)
    print("shape:", train_df.shape)

    print("Get texts, intersection...")

    # Step 6 - Remove questions interactions we do not have text
    texts_df = get_cloudacademy_texts(personal_cleaning=personal_cleaning, texts_filepath=texts_filepath, n_texts=n_texts, make_sentences_flag=make_sentences_flag)
    train_df = train_df.loc[train_df['problem_id'].isin(texts_df['problem_id'])]
    texts_df = texts_df.loc[texts_df['problem_id'].isin(train_df['problem_id'])]

    n_ids = len(questions_ids)
    # n_skills = len(train_df['skill'].unique())
    print("no. of problems :", n_ids)
    # print("no. of skills: ", n_skills)
    print("shape of intersection after intersection of interactions and texts datasets:", train_df.shape)
    train_df['skill'] = 0
    texts_df['question_id'], _ = pd.factorize(texts_df['problem_id'], sort=True)
    train_df['question_id'], _ = pd.factorize(train_df['problem_id'], sort=True)

    texts_df.to_csv(output_filepath+'texts_processed.csv')
    train_df.to_csv(output_filepath+'interactions_processed.csv')
    return train_df, texts_df
