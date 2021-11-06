import gc

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from code.models.encoding_models.count_vectorizer import count_vectorizer
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
    df = df[['user_id', 'problem_id', 'correct', 'skill']]
    print(df)
    # Step 3.1 - Generate NLP extracted encoding for problems
    encode_model = count_vectorizer(min_df=min_df, max_df=max_df, binary=False, max_features=max_features)
    encode_model.fit(text_df, save_filepath)

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
            for problem, label, skill in list(zip(group['problem_id'].values, group['correct'].values, group['skill'].values)):
                encoding = encode_model.get_encoding(problem)
                encoding = np.expand_dims(encoding, axis=0)
                document_to_term.append(encoding)
                labels = np.append(labels, label)
                skills = np.append(skills, skill)
            document_to_term = np.concatenate(document_to_term, axis=0)
            i_doc = document_to_term[:-1]
            o_doc = document_to_term[1:]
            i_label = labels[:-1]
            o_label = labels[1:]
            i_skill = skills[:-1]
            o_skill = skills[1:]
            inputs = (i_doc, i_skill, i_label)
            outputs = (o_doc, o_skill, o_label)
            yield inputs, outputs

    def generate_encodings_test():
        for name, group in df.loc[df['user_id'].isin(test_users)].groupby('user_id'):
            document_to_term = []
            labels = np.array([], dtype=np.int)
            for problem, label, skill in list(zip(group['problem_id'].values, group['correct'].values, group['skill'].values)):
                encoding = encode_model.get_encoding(problem)
                encoding = np.expand_dims(encoding, axis=0)
                document_to_term.append(encoding)
                labels = np.append(labels, label)
                skills = np.append(skills, skill)
            document_to_term = np.concatenate(document_to_term, axis=0)
            i_doc = document_to_term[:-1]
            o_doc = document_to_term[1:]
            i_label = labels[:-1]
            o_label = labels[1:]
            i_skill = skills[:-1]
            o_skill = skills[1:]
            inputs = (i_doc, i_skill, i_label)
            outputs = (o_doc, o_skill, o_label)
            yield inputs, outputs

    def generate_encodings_train():
        for name, group in df.loc[df['user_id'].isin(train_users)].groupby('user_id'):
            document_to_term = []
            labels = np.array([], dtype=np.int)
            for problem, label, skill in list(zip(group['problem_id'].values, group['correct'].values, group['skill'].values)):
                encoding = encode_model.get_encoding(problem)
                encoding = np.expand_dims(encoding, axis=0)
                document_to_term.append(encoding)
                labels = np.append(labels, label)
                skills = np.append(skills, skill)
            document_to_term = np.concatenate(document_to_term, axis=0)
            i_doc = document_to_term[:-1]
            o_doc = document_to_term[1:]
            i_label = labels[:-1]
            o_label = labels[1:]
            i_skill = skills[:-1]
            o_skill = skills[1:]
            inputs = (i_doc, i_skill, i_label)
            outputs = (o_doc, o_skill, o_label)
            yield inputs, outputs

    encoding_depth = encode_model.vector_size
    skill_depth = df['skill'].max() + 1

    def create_dataset(generate_encodings, users, encoding_depth, skill_depth):
        types = ((tf.float32, tf.int32, tf.float32),
                 (tf.float32, tf.int32, tf.float32))
        shapes = (([None, encoding_depth], [None], [None]),
                  ([None, encoding_depth], [None], [None]))
        # Step 5 - Get Tensorflow Dataset
        dataset = tf.data.Dataset.from_generator(
            generator=generate_encodings,
            output_types=types,
            output_shapes=shapes
        )

        nb_users = len(users)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=nb_users)

        dataset = dataset.map(
            lambda inputs, outputs: (
                (inputs[0], tf.one_hot(inputs[1], depth=skill_depth), tf.expand_dims(inputs[2], axis=-1)),
                tf.concat(values=[
                    outputs[0],
                    tf.one_hot(outputs[1], depth=skill_depth),
                    tf.expand_dims(outputs[2], axis=-1)],
                    axis=-1)
            )
        )

        # Step 6 - Encode categorical features and merge skills with labels to compute target loss.
        # More info: https://github.com/tensorflow/tensorflow/issues/32142


        # Step 7 - Pad sequences per batch
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padding_values=MASK_VALUE,
            drop_remainder=True
        )

        return dataset

    train_set = create_dataset(generate_encodings_train, train_users, encoding_depth, skill_depth)
    val_set = create_dataset(generate_encodings_val, val_users, encoding_depth, skill_depth)
    test_set = create_dataset(generate_encodings_test, test_users, encoding_depth, skill_depth)

    return train_set, val_set, test_set, encoding_depth, skill_depth


def get_target(y_true, y_pred, nb_encodings=300, nb_skills=300):
    # Get skills and labels from y_true

    mask = 1 - tf.cast(tf.equal(y_true, MASK_VALUE), y_true.dtype)
    y_true = y_true * mask
    encodings_true, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)
    encodings_pred, y_pred = tf.split(y_pred, num_or_size_splits=[-1, 1], axis=-1)

    # Get predictions for each skill

    return y_true, y_pred
