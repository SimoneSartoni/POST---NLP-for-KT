import pandas as pd
from Knowledge_Tracing.code.data_processing.preprocess.text_processing_utils import *


def load_process_poj_texts(personal_cleaning=True, texts_filepath='../input/', n_texts=None, make_sentences_flag=True):
    if n_texts:
        df = pd.read_csv(texts_filepath, low_memory=False, sep='\n', names=["data"], nrows=n_texts)
    else:
        df = pd.read_csv(texts_filepath, low_memory=False, sep='\n', names=["data"])
    questions = []
    problem_ids = []
    index = -1
    print(df["data"])
    for row in df["data"]:
        if '#' in row:
            array = row.split('#')
            if array[0].isdigit():
                questions.append(" ")
                index += 1
                id = int(array[0])
                problem_ids.append(id)
                questions[index] = array[1]

        else:
            new = str(questions[index]) + str(row)
            questions[index] = new
    questions.append([])
    dictionary = {'problem_id': problem_ids, 'body': questions}
    df = pd.Dataframe(dictionary)
    # Using the preprocessing function to preprocess the tweet data
    preprocess_data(df, 'body')
    print("df after preprocess data:")
    print(df)
    df['plain_text'] = df['body']
    # Using tokenizer and removing the stopwords
    rem_stopwords_tokenize(df, 'body', personal_cleaning)
    print("df after stopwords tokenize:")
    print(df)
    df['body'] = df['body'].apply(lambda x: remove_issues(x))
    print("df after personal cleaning:")
    print(df)
    # Converting all the texts back to sentences
    lemmatize_all(df, 'body')
    print("df after lemmatize all:")
    print(df)
    if make_sentences_flag:
        make_sentences(df, 'body')
    return df
