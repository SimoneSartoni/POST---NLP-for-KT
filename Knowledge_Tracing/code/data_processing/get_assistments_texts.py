import re

import pandas as pd

import nltk
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Knowledge_Tracing.code.data_processing.data_processing import remove_issues
import hunspell

# Function to preprocess the tweets data
def preprocess_data(data, name):
    # Lowering the case of the words in the sentences
    data[name] = data[name].str.lower()

    def clean_tags(text):
        text_with_clean_tags = ""
        if not pd.isnull(text):
            soup = BeautifulSoup(text, "html.parser")
            for el in soup.find_all('img'):
                text_with_clean_tags = text_with_clean_tags + ' ' + str(el.unwrap())
            p = re.compile(r'<.*?>')
            text_with_clean_tags = text_with_clean_tags + ' ' + p.sub('', text)
        return text_with_clean_tags

    data[name].apply(lambda x: clean_tags(x))
    print("after clean tags")
    print(data[name])
    # Code to remove the Hashtags from the text
    data[name] = data[name].apply(lambda x: re.sub(r'\B#\S+', '', str(x)))
    # Code to remove the links from the text
    data[name] = data[name].apply(lambda x: re.sub(r"http\S+", "", str(x)))
    # Code to remove the Special characters from the text
    data[name] = data[name].apply(lambda x: ' '.join(re.findall(r'\w+', str(x))))
    # Code to substitute the multiple spaces with single spaces
    data[name] = data[name].apply(lambda x: re.sub(r'\s+', ' ', str(x), flags=re.I))
    # Code to remove all the single characters in the text
    data[name] = data[name].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', '', str(x)))
    # Code to remove all digits from text:
    data[name] = data[name].apply(lambda x: re.sub(r'[0-9]+', '', str(x)))
    # Remove the twitter handlers
    data[name] = data[name].apply(lambda x: re.sub(r'@[^\s]+', '', str(x)))


# This function is to remove stopwords from a particular column and to tokenize it
def rem_stopwords_tokenize(data, name, personal_cleaning):
    stop_words = set(stopwords.words('english'))
    dictionary = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')

    def escape_values(text):
        text = text.replace(' ', '#').replace('/', '#slash#').replace('<', '#lessthan#').replace('>',
                                                                                                 '#morethan#').replace(
            ",", "#comma#").replace(";", "#semicolon#").replace(".", "#dot#").replace("?", "#questionmark#").replace(
            "!", "exclamationpoint").replace("=", "#equal#").replace("\\", "#").replace("%", "#percentage#").replace(
            "\\t", "#").replace("\\n", "#").replace("\t", "#").replace("\n", "#").replace('\"', "##").replace(
            "(", "#openroundbracket#").replace(")", "#closeroundbracket#").replace("[", "#opensquarebracket#").replace(
            "]", "#closesquarebracket#").replace("_", "#underscore#").replace("&", "#ampersand#"). \
            replace("}", "#closebrace#").replace("{", "#openbrace#").replace("+", "#plus#").replace("-", "#minus#"). \
            replace("*", "#multiplication#").replace("€", "#euros#").replace("$", "#dollar#"). \
            replace("^", "#powerof#exponent#").replace(":", "#colon#")

        text = str(text).replace('#', ' ')
        return text

    def getting(sen):
        example_sent = sen
        word_tokens = word_tokenize(example_sent)
        filtered_sentence = list(set(word_tokens).difference(stop_words))
        return list(filtered_sentence)

    def filter_existing_words(text):
        filtered_text = []
        for word in text:
            if dictionary.spell(word):
                filtered_text.append(word)
        return filtered_text

    data[name] = data[name].apply(lambda text: escape_values(text))
    data[name] = data[name].apply(lambda text: getting(text))
    data[name] = data[name].apply(lambda text: filter_existing_words(text))


# Making a function to lemmatize all the words
lemmatizer = WordNetLemmatizer()


def lemmatize_all(data, name):
    arr = data[name]
    a = []
    for i in arr:
        b = []
        for j in i:
            x = lemmatizer.lemmatize(j, pos='a')
            b.append(x)
        a.append(b)
    data[name] = a


# Function to make it back into a sentence
def make_sentences(data, name):
    data[name] = data[name].apply(lambda x: ' '.join([i + ' ' for i in x]))
    # Removing double spaces if created
    data[name] = data[name].apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.I))


def get_assistments_texts(personal_cleaning=True, texts_filepath='../input/', n_texts=None, make_sentences_flag=True):
    input_types = {'problem_id': 'int64', 'body': "string"}
    if n_texts:
        df = pd.read_csv(texts_filepath, low_memory=False, dtype=input_types, nrows=n_texts)
    else:
        df = pd.read_csv(texts_filepath, low_memory=False, dtype=input_types)
    # Using the preprocessing function to preprocess the tweet data
    preprocess_data(df, 'body')
    print("df after preprocess data:")
    print(df)
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
