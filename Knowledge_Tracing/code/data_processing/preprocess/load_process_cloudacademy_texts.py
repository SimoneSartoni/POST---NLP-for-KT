import pandas as pd
import hunspell

from Knowledge_Tracing.code.data_processing.preprocess.text_processing_utils import *


def get_cloudacademy_texts(personal_cleaning=True, texts_filepath='../input/', n_texts=None, make_sentences_flag=False):
    input_types = {'id': 'int64', 'description': "string"}
    if n_texts:
        df = pd.read_csv(texts_filepath, low_memory=False, dtype=input_types, nrows=n_texts)
    else:
        df = pd.read_csv(texts_filepath, low_memory=False, dtype=input_types)
    # Using the preprocessing function to preprocess the tweet data
    renaming_dict = {"id": "problem_id", "description": "body"}
    df = df.rename(columns=renaming_dict, errors="raise")
    df['unprocessed_text'] = df['body']
    print("df after hunspell")
    dictionary_US = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff',  format="html")
    df['rapid'] = df['body'].apply(lambda text: dictionary_US.spell(text))
    print(df)
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
