import re

import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Function to preprocess the tweets data
def preprocess_data(data, name):
    # Lowering the case of the words in the sentences
    data[name] = data[name].str.lower()
    print(data[name])
    # Code to remove the Hashtags from the text
    data[name]=data[name].apply(lambda x: re.sub(r'\B#\S+', '', str(x)))
    # Code to remove the links from the text
    data[name]=data[name].apply(lambda x: re.sub(r"http\S+", "", str(x)))
    # Code to remove the Special characters from the text
    data[name]=data[name].apply(lambda x: ' '.join(re.findall(r'\w+', str(x))))
    # Code to substitute the multiple spaces with single spaces
    data[name]=data[name].apply(lambda x: re.sub(r'\s+', ' ', str(x), flags=re.I))
    # Code to remove all the single characters in the text
    data[name]=data[name].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', '', str(x)))
    # Remove the twitter handlers
    data[name]=data[name].apply(lambda x: re.sub(r'@[^\s]+', '', str(x)))


# This function is to remove stopwords from a particular column and to tokenize it
def rem_stopwords_tokenize(data, name):
    def getting(sen):
        example_sent = sen
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(example_sent)
        filtered_sentence = []

        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w.lower())
        return filtered_sentence

    x = []
    for i in data[name].values:
        x.append(getting(i))
    data[name] = x


# Making a function to lemmatize all the words
lemmatizer = WordNetLemmatizer()


def lemmatize_all(data, name):
    arr = data[name]
    a = []
    for i in arr:
        b = []
        for j in i:
            x = lemmatizer.lemmatize(j, pos='a')
            x = lemmatizer.lemmatize(x)
            b.append(x)
        a.append(b)
    data[name] = a


# Function to make it back into a sentence
def make_sentences(data, name):

    data[name]=data[name].apply(lambda x: ' '.join([i+' ' for i in x]))
    # Removing double spaces if created
    data[name]=data[name].apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.I))


def get_assistments_texts(texts_filepath='../input/'):
    input_types = {'problem_id': 'int64', 'body': "string"}
    df = pd.read_csv(texts_filepath, low_memory=False, dtype=input_types)
    # Using the preprocessing function to preprocess the tweet data
    print(df['body'].values[0:10])
    preprocess_data(df, 'body')
    # Using tokenizer and removing the stopwords
    rem_stopwords_tokenize(df, 'body')
    # Converting all the texts back to sentences
    make_sentences(df, 'body')
    print(df['body'].values[0:10])
    return df
