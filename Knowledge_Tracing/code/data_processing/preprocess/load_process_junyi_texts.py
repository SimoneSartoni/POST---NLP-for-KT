from Knowledge_Tracing.code.data_processing.preprocess.text_processing_utils import *
import pandas as pd


def load_process_junyi_texts(personal_cleaning=True, texts_filepath='../input/', n_texts=None, make_sentences_flag=True):
    if n_texts:
        df = pd.read_csv(texts_filepath, low_memory=False, sep='#')
    else:
        df = pd.read_csv(texts_filepath, low_memory=False, sep='#')
    # Using the preprocessing function to preprocess the tweet data
    df['question_name'], df['chinese_question'], df[
        'chinese_question_desc']
    df['body'] = df['question_name'] + df['chinese_question'] + df['chinese_question_desc']
    df['problem_id'], _ = pd.factorize(df['question_name'], sort=False)
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
