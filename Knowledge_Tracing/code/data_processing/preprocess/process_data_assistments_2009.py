from datetime import datetime

import pandas as pd


from Knowledge_Tracing.code.utils.utils import try_parsing_date
from Knowledge_Tracing.code.data_processing.preprocess.load_process_assistments_texts import load_process_assistments_texts


def process_data_assistments_2009(min_questions=2, max_questions=50,
                                  interactions_filepath="",
                                  texts_filepath='', output_filepath="", n_rows=None,
                                  n_texts=None, personal_cleaning=True, make_sentences_flag=True):
    dtypes = {'user_id': 'int32', 'problem_id': 'int64',
              'correct': 'float64', 'skill_id': "string",
              'order_id': "string"}

    dataset_info = {}
    print("loading csv.....")
    if n_rows:
        train_df = pd.read_csv(interactions_filepath, dtype=dtypes, nrows=n_rows, encoding='unicode_escape')
    else:
        train_df = pd.read_csv(interactions_filepath, dtype=dtypes, encoding='unicode_escape')
    print("shape of dataframe :", train_df.shape)
    questions_ids = train_df['problem_id'].unique()
    n_ids = len(questions_ids)
    n_skills = len(train_df['skill_id'].unique())
    print("no. of problems :", n_ids)
    print("no. of skills: ", n_skills)
    print("shape:", train_df.shape)
    print("number of users:" + str(len(train_df['user_id'].unique())))

    # Step 3.1 - Define start, end and elapsed time, fill no timed elapsed time and cap values under a max
    train_df['order_id'] = [try_parsing_date(x) for x in train_df['order_id']]

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

    # Step 2.1 - Fill no skilled question with "no_skill" token
    train_df.fillna("no_skill", inplace=True)
    print("shape after drop no skill:", train_df.shape)

    # Step 2.2 - Enumerate skill ids and question ids
    train_df['skill'], _ = pd.factorize(train_df['skill_id'], sort=True)
    print("shape after factorize:", train_df.shape)

    # Step 5 - Compute number of unique skills ids and number of unique question ids
    questions_ids = train_df['problem_id'].unique()
    n_ids = len(questions_ids)
    n_skills = len(train_df['skill'].unique())
    print("no. of problems :", n_ids)
    print("no. of skills: ", n_skills)
    print("shape after exclusion:", train_df.shape)
    print("number of users:" + str(len(train_df['user_id'].unique())))

    print("Get texts, intersection...")

    # Step 6 - Remove questions interactions we do not have text
    texts_df = load_process_assistments_texts(personal_cleaning=personal_cleaning, texts_filepath=texts_filepath,
                                              n_texts=n_texts, make_sentences_flag=make_sentences_flag)
    train_df = train_df.loc[train_df['problem_id'].isin(texts_df['problem_id'])]

    texts_df = texts_df.loc[texts_df['problem_id'].isin(train_df['problem_id'])]

    questions_ids = train_df['problem_id'].unique()
    n_ids = len(questions_ids)
    n_skills = len(train_df['skill'].unique())
    print("no. of problems :", n_ids)
    print("no. of skills: ", n_skills)
    print("shape after exclusion:", train_df.shape)
    print("number of users:" + str(len(train_df['user_id'].unique())))

    train_df['skill'], _ = pd.factorize(train_df['skill'], sort=True)
    texts_df['question_id'], _ = pd.factorize(texts_df['problem_id'], sort=True)
    train_df['question_id'], _ = pd.factorize(train_df['problem_id'], sort=True)

    texts_df.to_csv(output_filepath + 'texts_processed.csv')
    train_df.to_csv(output_filepath + 'interactions_processed.csv')
    return train_df, texts_df
