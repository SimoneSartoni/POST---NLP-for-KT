from datetime import datetime

import numpy as np
import pandas as pd

from Knowledge_Tracing.code.utils.utils import try_parsing_date
from Knowledge_Tracing.code.data_processing.get_assistments_texts import get_assistments_texts
from ast import literal_eval

def load_preprocessed_interactions(interactions_filepath=""):
    dtypes = {'user_id': 'int32', 'problem_id': 'int64',
              'correct': 'float64',
              'start_time': "string", 'end_time': "string",
              'skill': "int32", 'elapsed_time': 'int64',
              'timestamp': "string", 'question_id': "int64"}
    print("loading csv.....")
    train_df = pd.read_csv(interactions_filepath, dtype=dtypes)
    print("shape of dataframe :", train_df.shape)
    return train_df


def load_preprocessed_texts(texts_filepath=""):
    dtypes = {'problem_id': 'int64', 'body': "string", 'question_id': "int64"}
    print("loading csv.....")
    texts_df = pd.read_csv(texts_filepath, dtype=dtypes)
    texts_df['body'] = texts_df['body'].apply(lambda x: literal_eval(x))
    print("shape of dataframe :", texts_df.shape)
    return texts_df
