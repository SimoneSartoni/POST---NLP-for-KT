import pandas as pd

from Knowledge_Tracing.code.utils.utils import try_parsing_date
from Knowledge_Tracing.code.data_processing.preprocess.load_process_cloudacademy_texts import get_cloudacademy_texts


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
    dtype = {'_actor_id': 'string', 'session_mode': 'string', '_time_stamp': 'string', 'session_step': 'int32',
             'timer': 'int32', 'elapsed_time': 'int32', 'action': 'string', 'correct': 'string', 'question_id': 'int64',
             'session_id': 'int64', 'source': 'string', '_platform': 'string', 'certification_id': 'float64'}
    print("loading csv.....")
    if n_rows:
        train_df = pd.read_csv(interactions_filepath, names=input_columns, dtype=dtype, nrows=n_rows)
    else:
        train_df = pd.read_csv(interactions_filepath, names=input_columns, dtype=dtype)
    print("shape of dataframe :", train_df.shape)
    print(train_df)
    renaming_dict = {"_actor_id": "user_id", "_time_stamp": "timestamp", "question_id": "problem_id", }
    train_df = train_df.rename(columns=renaming_dict, errors="raise")


    print("removing session_mode different from test or exam")
    train_df = train_df.loc[train_df['session_mode'].isin(['test', 'exam'])]
    print("shape after exclusion:", train_df.shape)
    print(train_df)
    # Step 4 - Sort interactions according to timestamp
    train_df['timestamp'] = [try_parsing_date(x) for x in train_df['timestamp']]
    train_df = train_df.sort_values(["timestamp"], ascending=True)

    train_df = train_df.drop_duplicates(subset=['user_id', 'problem_id'], keep='first').reset_index(drop=True)

    # Step 1 - Remove users with less than a certain number of answers
    # train_df = train_df.groupby('user_id').tail(max_questions)
    print("shape after drop duplicated interactions:", train_df.shape)

    # Step 1 - Remove users with less than a certain number of answers
    train_df = train_df.groupby('user_id').filter(lambda q: len(q) >= min_questions).copy()
    print("shape after at least " + str(min_questions) + " interactions:", train_df.shape)

    # Step 2.1 - Fill no skilled question with "no_skill" token
    train_df.fillna("NA", inplace=True)
    print("shape after filling NA values:", train_df.shape)

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
    texts_df = get_cloudacademy_texts(personal_cleaning=personal_cleaning, texts_filepath=texts_filepath,
                                      n_texts=n_texts, make_sentences_flag=make_sentences_flag)
    train_df = train_df.loc[train_df['problem_id'].isin(texts_df['problem_id'])]
    texts_df = texts_df.loc[texts_df['problem_id'].isin(train_df['problem_id'])]

    train_df['correct'] = [1.0 if answer == 't' else 0.0 if answer == 'f' else 0.0 for answer in train_df['correct']]
    print(train_df['correct'].mean())
    print(train_df.describe())
    n_ids = len(questions_ids)
    # n_skills = len(train_df['skill'].unique())
    print("no. of problems :", n_ids)
    # print("no. of skills: ", n_skills)
    print("shape of intersection after intersection of interactions and texts datasets:", train_df.shape)
    train_df['skill'] = 0
    texts_df['question_id'], _ = pd.factorize(texts_df['problem_id'], sort=True)
    train_df['question_id'], _ = pd.factorize(train_df['problem_id'], sort=True)

    texts_df.to_csv(output_filepath + 'texts_processed.csv')
    train_df.to_csv(output_filepath + 'interactions_processed.csv')
    return train_df, texts_df
