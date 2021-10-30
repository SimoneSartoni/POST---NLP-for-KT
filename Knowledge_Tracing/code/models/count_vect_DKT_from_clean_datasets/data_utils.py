import gc

import pandas as pd
import tensorflow as tf
import numpy as np
from Knowledge_Tracing.code.data_processing.dataset import dataset as dt
from Knowledge_Tracing.code.evaluation.predictors.logistic_regression import logistic_regressor
from Knowledge_Tracing.code.models.count_vectorizer.count_vectorizer import count_vectorizer
from Knowledge_Tracing.code.data_processing.get_data_assistments_2012 import get_data_assistments_2012
from Knowledge_Tracing.code.data_processing.get_data_assistments_2009 import get_data_assistments_2009

MASK_VALUE = -1.0  # The masking value cannot be zero.


def load_dataset(batch_size=32, shuffle=True, dataset_name='assistment_2012',
                 interactions_filepath="../input/assistmentds-2012/2012-2013-data-with-predictions-4-final"
                                       ".csv",
                 save_filepath='/kaggle/working/', texts_filepath='../input/', min_df=2, max_df=1.0,
                 min_questions=2, max_features=1000, max_questions=25, n_rows=None, n_texts=None,
                 personal_cleaning=True):
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
    # Step 3.1 - Generate NLP extracted encoding for problems
    encode_model = count_vectorizer(min_df=min_df, max_df=max_df, binary=False, max_features=max_features)
    encode_model.fit(text_df, save_filepath)

    max_value = encode_model.words_num
    del text_df
    gc.collect()
    print("number of words is: " + str(max_value))

    def generate_encodings():
        for name, group in df.groupby('user_id'):
            print(name)
            print(group)
            document_to_term = []
            labels = np.array([], dtype=np.int)
            for problem, label in list(zip(group['problem_id'].values, group['correct'].values)):
                print(problem)
                encoding = encode_model.get_encoding(problem)
                encoding = np.expand_dims(encoding, axis=0)
                document_to_term.append(encoding)
                labels = np.append(labels, label)
            document_to_term = np.concatenate(document_to_term, axis=0)
            i_doc = document_to_term[:-1]
            o_doc = document_to_term[1:]
            i_label = labels[:-1]
            o_label = labels[1:]
            inputs = (i_doc, i_label)
            outputs = (o_doc, o_label)
            yield inputs, outputs

    encoding_depth = encode_model.vector_size

    types = ((tf.float32, tf.float32),
             (tf.float32, tf.float32))
    shapes = (([None, encode_model.vector_size], [None]),
              ([None, encode_model.vector_size], [None]))
    # Step 5 - Get Tensorflow Dataset
    dataset = tf.data.Dataset.from_generator(
        generator=generate_encodings,
        output_types=types,
        output_shapes=shapes
    )
    nb_users = len(df.groupby('user_id'))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=nb_users)

    print(dataset)
    dataset = dataset.map(
        lambda inputs, outputs: (
            (inputs[0], tf.expand_dims(inputs[1], axis=-1)),
            tf.concat(values=[
                outputs[0],
                tf.expand_dims(outputs[1], axis=-1)],
                axis=-1)
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

    length = nb_users // batch_size
    return dataset, length, encoding_depth


def get_target(y_true, y_pred, nb_encodings=300):
    # Get skills and labels from y_true

    mask = 1 - tf.cast(tf.equal(y_true, MASK_VALUE), y_true.dtype)
    y_true = y_true * mask
    encodings_true, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)
    encodings_pred, y_pred = tf.split(y_pred, num_or_size_splits=[-1, 1], axis=-1)

    # Get predictions for each skill

    return y_true, y_pred
