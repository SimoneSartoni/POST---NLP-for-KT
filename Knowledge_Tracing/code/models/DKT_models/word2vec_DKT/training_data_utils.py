import gc

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from Knowledge_Tracing.code.models.encoding_models.gensim_model.gensim_word2vec import word2vec
from Knowledge_Tracing.code.data_processing.get_data_assistments_2012 import get_data_assistments_2012
from Knowledge_Tracing.code.data_processing.get_data_assistments_2009 import get_data_assistments_2009

MASK_VALUE = -1.  # The masking value cannot be zero.


def load_dataset(batch_size=32, shuffle=True, dataset_name='assistment_2012',
                 interactions_filepath="../input/assistmentds-2012/2012-2013-data-with-predictions-4-final"
                                       ".csv",
                 save_filepath='/kaggle/working/', texts_filepath='../input/',
                 min_questions=2, max_questions=25, n_rows=None, n_texts=None, personal_cleaning=True,
                 min_df=2, max_df=1.0, max_features=1000,
                 keyed_vectors="", normalize_encoding=False):
    if dataset_name == 'assistment_2012':
        df, text_df = get_data_assistments_2012(min_questions=min_questions, max_questions=max_questions,
                                                interactions_filepath=interactions_filepath,
                                                texts_filepath=texts_filepath, n_rows=n_rows, n_texts=n_texts,
                                                make_sentences_flag=False, personal_cleaning=personal_cleaning)
    elif dataset_name == 'assistment_2009':
        df, text_df = get_data_assistments_2009(min_questions=min_questions, max_questions=max_questions,
                                                interactions_filepath=interactions_filepath,
                                                texts_filepath=texts_filepath, n_rows=n_rows, n_texts=n_texts,
                                                make_sentences_flag=False, personal_cleaning=personal_cleaning, )

    print(df)
    df = df[['user_id', 'problem_id', 'correct']]
    print(df)

    if max_features:
        encode_model = word2vec(min_count=min_df, vector_size=max_features)
    else:
        encode_model = word2vec(min_count=min_df)
    encode_model.encode_problems(texts_df=text_df)
    encode_model.fit()


    max_value = encode_model.words_num
    del text_df
    gc.collect()
    print("number of words is: " + str(max_value))
    users = df['user_id'].unique()
    train_users, test_users = train_test_split(users, test_size=0.2)
    train_users, val_users = train_test_split(train_users, test_size=0.2)

    def generate_encodings_val():
        for name, group in df.loc[df['user_id'].isin(val_users)].groupby('user_id'):
            document_to_term = []
            labels = np.array([], dtype=np.int)
            for problem, label in list(zip(group['problem_id'].values, group['correct'].values)):
                encoding = encode_model.get_encoding(problem, norm=normalize_encoding)
                zeros = np.zeros(encoding.shape, dtype=np.float)
                if label:
                    encoding = np.concatenate([encoding, zeros])
                else:
                    encoding = np.concatenate([zeros, encoding])
                encoding = np.expand_dims(encoding, axis=0)
                document_to_term.append(encoding)
                labels = np.append(labels, label)
            document_to_term = np.concatenate(document_to_term, axis=0)
            i_doc = document_to_term[:-1]
            o_label = labels[1:]
            inputs = i_doc
            outputs = o_label
            yield inputs, outputs

    def generate_encodings_test():
        for name, group in df.loc[df['user_id'].isin(test_users)].groupby('user_id'):
            document_to_term = []
            labels = np.array([], dtype=np.int)
            for problem, label in list(zip(group['problem_id'].values, group['correct'].values)):
                encoding = encode_model.get_encoding(problem, norm=normalize_encoding)
                zeros = np.zeros(encoding.shape, dtype=np.float)
                if label:
                    encoding = np.concatenate([encoding, zeros])
                else:
                    encoding = np.concatenate([zeros, encoding])
                encoding = np.expand_dims(encoding, axis=0)
                document_to_term.append(encoding)
                labels = np.append(labels, label)
            document_to_term = np.concatenate(document_to_term, axis=0)
            i_doc = document_to_term[:-1]
            o_label = labels[1:]
            inputs = i_doc
            outputs = o_label
            yield inputs, outputs

    def generate_encodings_train():
        for name, group in df.loc[df['user_id'].isin(train_users)].groupby('user_id'):
            document_to_term = []
            labels = np.array([], dtype=np.int)
            for problem, label in list(zip(group['problem_id'].values, group['correct'].values)):
                encoding = encode_model.get_encoding(problem, norm=normalize_encoding)
                zeros = np.zeros(encoding.shape, dtype=np.float)
                if label:
                    encoding = np.concatenate([encoding, zeros])
                else:
                    encoding = np.concatenate([zeros, encoding])
                encoding = np.expand_dims(encoding, axis=0)
                document_to_term.append(encoding)
                labels = np.append(labels, label)
            document_to_term = np.concatenate(document_to_term, axis=0)
            i_doc = document_to_term[:-1]
            o_label = labels[1:]
            inputs = i_doc
            outputs = o_label
            yield inputs, outputs

    encoding_depth = 2 * encode_model.vector_size

    def create_dataset(generate_encodings, users, encoding_depth):
        # Step 5 - Get Tensorflow Dataset
        types = (tf.float32,
                 tf.float32)
        shapes = ([None, encoding_depth], [None])
        # Step 5 - Get Tensorflow Dataset
        dataset = tf.data.Dataset.from_generator(
            generator=generate_encodings,
            output_types=types,
            output_shapes=shapes
        )

        nb_users = len(users)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=nb_users, reshuffle_each_iteration=True)

        print(dataset)
        dataset = dataset.map(
            lambda inputs, outputs: (
                inputs,
                tf.expand_dims(outputs, axis=-1)
            )
        )

        # Step 6 - Encode categorical features and merge skills with labels to compute target loss.
        # More info: https://github.com/tensorflow/tensorflow/issues/32142

        print(dataset)

        # Step 7 - Pad sequences per batch
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padding_values=MASK_VALUE,
            drop_remainder=True
        )

        print(dataset)
        return dataset

    train_set = create_dataset(generate_encodings_train, train_users, encoding_depth)
    val_set = create_dataset(generate_encodings_val, val_users, encoding_depth)
    test_set = create_dataset(generate_encodings_test, test_users, encoding_depth)

    return train_set, val_set, test_set, encoding_depth


def get_target(y_true, y_pred, nb_encodings=300):
    # Get skills and labels from y_true

    mask = 1 - tf.cast(tf.equal(y_true, MASK_VALUE), y_true.dtype)
    y_true = y_true * mask
    encodings_true, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)
    encodings_pred, y_pred = tf.split(y_pred, num_or_size_splits=[-1, 1], axis=-1)

    # Get predictions for each skill

    return y_true, y_pred