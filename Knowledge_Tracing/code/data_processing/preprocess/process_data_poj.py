from datetime import datetime

from Knowledge_Tracing.code.utils.utils import try_parsing_date
from Knowledge_Tracing.code.data_processing.preprocess.load_process_poj_texts import load_process_poj_texts
import pandas as pd


def process_data_poj(min_questions=2, max_questions=50,
                                  interactions_filepath="../input/assistmentds-2012/2012-2013-data-with-predictions-4"
                                                        "-final.csv",
                                  texts_filepath='../input/', output_filepath="/kaggle/working/", n_rows=None,
                                  n_texts=None, personal_cleaning=True, make_sentences_flag=True):
    dtypes = {'User': 'int32', 'Problem': 'int64',
              'Result': 'string',
              'Submit Time': "string"}
    print("loading csv.....")
    if n_rows:
        train_df = pd.read_csv(interactions_filepath, nrows=n_rows)
    else:
        train_df = pd.read_csv(interactions_filepath,)
    print("shape of dataframe :", train_df.shape)
    renaming_dict = {"Problem": "problem_id", "User": "user_id", "Result": "correct", "Submit Time": "order_id"}
    train_df = train_df.rename(columns=renaming_dict, errors="raise")
    # Step 3.1 - Define start, end and elapsed time, fill no timed elapsed time and cap values under a max
    train_df['order_id'] = [try_parsing_date(x) for x in train_df['order_id']]
    train_df['correct'] = [1.0 if c == "Accepted" else 0.0 for c in train_df['correct']]
    # Step 3.2 - Generate timestamps from start time
    train_df["timestamp"] = [datetime.strptime(start, '%Y-%m-%d %H:%M:%S').timestamp()
                             for start in train_df['order_id']]

    # Step 4 - Sort interactions according to timestamp
    train_df = train_df.sort_values(["timestamp"], ascending=True).reset_index(drop=True)

    # Step 1 - Remove users with less than a certain number of answers
    train_df = train_df.groupby('user_id').tail(max_questions)
    print("shape after at max max " + str(max_questions) + " interactions:", train_df.shape)

    train_df = train_df.drop_duplicates(subset=['user_id', 'problem_id'], keep='first').reset_index(drop=True)

    # Step 1 - Remove users with less than a certain number of answers
    train_df = train_df.groupby('user_id').filter(lambda q: len(q) >= min_questions).copy()
    print("shape after at least 2 interactions:", train_df.shape)

    print("Get texts, intersection...")
    # Step 6 - Remove questions interactions we do not have text
    texts_df = load_process_poj_texts(personal_cleaning=personal_cleaning, texts_filepath=texts_filepath,
                                              n_texts=n_texts, make_sentences_flag=make_sentences_flag)
    train_df = train_df.loc[train_df['problem_id'].isin(texts_df['problem_id'])]

    texts_df = texts_df.loc[texts_df['problem_id'].isin(train_df['problem_id'])]

    questions_ids = train_df['problem_id'].unique()
    n_ids = len(questions_ids)
    print("no. of problems :", n_ids)
    print("shape after exclusion:", train_df.shape)
    texts_df['question_id'], _ = pd.factorize(texts_df['problem_id'], sort=True)
    train_df['question_id'], _ = pd.factorize(train_df['problem_id'], sort=True)

    texts_df.to_csv(output_filepath + 'texts_processed.csv')
    train_df.to_csv(output_filepath + 'interactions_processed.csv')
    return train_df, texts_df
