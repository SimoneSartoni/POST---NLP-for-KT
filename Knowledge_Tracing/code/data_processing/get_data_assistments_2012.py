import os
from datetime import datetime

import numpy as np
import pandas as pd
import psutil
import gc
from sklearn.metrics import roc_auc_score, accuracy_score

from torch.nn.utils.rnn import pad_sequence

import torch

from Knowledge_Tracing.code.utils.utils import try_parsing_date
from Knowledge_Tracing.code.data_processing.dataset import dataset as dt


def get_data_assistments_2012(min_questions=2, max_questions=50, interactions_filepath="../input/assistmentds-2012/2012-2013-data-with-predictions-4-final.csv",
                              texts_filepath='../input/'):
    dtypes = {'user_id': 'int32', 'problem_id': 'int64',
              'correct': 'float64', 'skill': "string",
              'start_time': "string", 'end_time': "string"}

    print("loading csv.....")

    train_df = pd.read_csv(interactions_filepath, dtype=dtypes)
    print("shape of dataframe :", train_df.shape)

    # Step 3.1 - Define start, end and elapsed time, fill no timed elapsed time and cap values under a max
    train_df['start_time'] = [try_parsing_date(x) for x in train_df['start_time']]
    train_df['end_time'] = [try_parsing_date(x) for x in train_df['end_time']]
    train_df["elapsed_time"] = [datetime.strptime(end, '%Y-%m-%d %H:%M:%S').timestamp() -
                                datetime.strptime(start, '%Y-%m-%d %H:%M:%S').timestamp()
                                for start, end in
                                list(zip(train_df['start_time'], train_df['end_time']))]
    train_df["elapsed_time"].fillna(300, inplace=True)
    train_df["elapsed_time"].clip(lower=0, upper=300, inplace=True)
    train_df["elapsed_time"] = train_df["elapsed_time"].astype(np.int)

    # Step 3.2 - Generate timestamps from start time
    train_df["timestamp"] = [datetime.strptime(start, '%Y-%m-%d %H:%M:%S').timestamp()
                             for start in train_df['start_time']]

    # Step 4 - Sort interactions according to timestamp
    train_df = train_df.sort_values(["timestamp"], ascending=True).reset_index(drop=True)

    # Step 1 - Remove users with less than a certain number of answers
    train_df = train_df.groupby('user_id').tail(max_questions)
    print("shape after at least 2 interactions:", train_df.shape)

    # Step 1 - Remove users with less than a certain number of answers
    train_df = train_df.groupby('user_id').filter(lambda q: len(q) >= min_questions).copy()
    print("shape after at least 2 interactions:", train_df.shape)

    # Step 2.1 - Fill no skilled question with "no_skill" token
    train_df.fillna("no_skill", inplace=True)
    print("shape after drop no skill:", train_df.shape)

    # Step 2.2 - Enumerate skill ids and question ids
    train_df['skill'], _ = pd.factorize(train_df['skill'], sort=True)
    train_df['question_id'], _ = pd.factorize(train_df['problem_id'], sort=True)
    print("shape after factorize:", train_df.shape)



    # Step 5 - Compute number of unique skills ids and number of unique question ids
    questions_ids = train_df['problem_id'].unique()
    n_ids = len(questions_ids)
    n_skills = len(train_df['skill'].unique())
    print("no. of problems :", n_ids)
    print("no. of skills: ", n_skills)
    print("shape after exclusion:", train_df.shape)

    # Step 6 - Remove questions interactions we do not have text
    loaded_dataset = dt(name="assistments_2012", path=texts_filepath, prefix="clean_datasets/")
    loaded_dataset.load_saved_texts()
    train_df = train_df.loc[train_df['problem_id'].isin(loaded_dataset.problem_id_to_index)]

    return train_df, loaded_dataset
