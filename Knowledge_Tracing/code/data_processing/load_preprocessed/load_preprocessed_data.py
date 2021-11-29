from datetime import datetime

import numpy as np
import pandas as pd

from ast import literal_eval


def load_preprocessed_interactions(interactions_filepath="", dictionary=None):
    dtypes = {'user_id': 'int32', 'problem_id': 'int64',
                'correct': 'float64', 'start_time':'string', 'end_time':'string',
                'skill': "int32", 'elapsed_time': 'int64',
                'timestamp': "string", 'question_id': "int64"}
    if dictionary:
        train_df = pd.read_csv(interactions_filepath, dtype=dictionary)
        for key in dtypes.keys():
            if key not in dictionary:
                train_df[key] = 0.0
    else:
        train_df = pd.read_csv(interactions_filepath, dtype=dtypes)
    print("loading csv.....")
    print("shape of dataframe :", train_df.shape)
    return train_df


def load_preprocessed_texts(texts_filepath="", text_as_sentence=False):
    dtypes = {'problem_id': 'int64', 'body': "string", 'question_id': "int64"}
    print("loading csv.....")
    texts_df = pd.read_csv(texts_filepath, dtype=dtypes)
    if not text_as_sentence:
        texts_df['body'] = texts_df['body'].apply(lambda x: literal_eval(x))
    texts_df['body'] = texts_df['body'].fillna("no_text", inplace=True)
    print("shape of dataframe :", texts_df.shape)
    return texts_df
