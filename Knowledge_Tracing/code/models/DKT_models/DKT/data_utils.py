import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

MASK_VALUE = -1.  # The masking value cannot be zero.
from Knowledge_Tracing.code.data_processing.get_data_assistments_2012 import get_data_assistments_2012
from Knowledge_Tracing.code.data_processing.get_data_assistments_2009 import get_data_assistments_2009


def load_dataset(batch_size=32, shuffle=True, dataset_name='assistment_2012',
                 interactions_filepath="../input/assistmentds-2012/2012-2013-data-with-predictions-4-final"
                                       ".csv",
                 save_filepath='/kaggle/working/', texts_filepath='../input/', min_df=2, max_df=1.0,
                 min_questions=2, max_features=1000, max_questions=25, n_rows=None, n_texts=None,
                 personal_cleaning=True):
    if dataset_name == 'assistment_2012':
        df, loaded_dataset = get_data_assistments_2012(min_questions=min_questions, max_questions=max_questions,
                                                       interactions_filepath=interactions_filepath,
                                                       texts_filepath=texts_filepath, n_rows=n_rows, n_texts=n_texts,
                                                       make_sentences_flag=False, personal_cleaning=personal_cleaning)
    elif dataset_name == 'assistment_2009':
        df, loaded_dataset = get_data_assistments_2009(min_questions=min_questions, max_questions=max_questions,
                                                       interactions_filepath=interactions_filepath,
                                                       texts_filepath=texts_filepath, n_rows=n_rows, n_texts=n_texts,
                                                       make_sentences_flag=False, personal_cleaning=personal_cleaning)
    # Step 3 - Cross skill id with answer to form a synthetic feature
    df['skill_with_answer'] = df['skill'] * 2 + df['correct']
    df['skill_with_answer'] = df['skill_with_answer'].astype('int32')
    print(df['skill_with_answer'])

    df = df[['user_id', 'problem_id', 'correct', 'skill_with_answer', 'skill']]

    # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    seq = df.groupby('user_id').apply(
        lambda r: (
            r['skill_with_answer'].values[:-1],
            r['skill'].values[1:],
            r['correct'].values[1:],
        )
    )
    nb_users = len(seq)

    train_seq, test_seq, val_seq = split_dataset(seq, nb_users, 0.2, 0.2)

    features_depth = df['skill_with_answer'].max() + 1
    skill_depth = df['skill'].max() + 1

    def create_dataset(seq, features_depth, skill_depth):
        # Step 5 - Get Tensorflow Dataset
        dataset = tf.data.Dataset.from_generator(
            generator=lambda: seq,
            output_types=(tf.int32, tf.int32, tf.float32)
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(seq), reshuffle_each_iteration=True)

        # Step 6 - Encode categorical features and merge skills with labels to compute target loss.
        # More info: https://github.com/tensorflow/tensorflow/issues/32142


        dataset = dataset.map(
            lambda feat, skill, label: (
                tf.one_hot(feat, depth=features_depth),
                tf.concat(
                    values=[
                        tf.one_hot(skill, depth=skill_depth),
                        tf.expand_dims(label, -1)
                    ],
                    axis=-1
                )
            )
        )

        # Step 7 - Pad sequences per batch
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padding_values=(MASK_VALUE, MASK_VALUE),
            padded_shapes=([None, None], [None, None]),
            drop_remainder=True
        )
        return dataset

    train_set = create_dataset(train_seq, features_depth, skill_depth)
    val_set = create_dataset(val_seq, features_depth, skill_depth)
    test_set = create_dataset(test_seq, features_depth, skill_depth)

    return train_set, val_set, test_set, features_depth, skill_depth


def split_dataset(sequences, total_size, test_fraction, val_fraction=None):
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between (0, 1)")

    if val_fraction is not None and not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between (0, 1)")

    train_set, test_set = train_test_split(sequences, test_size=test_fraction)

    val_set = None
    if val_fraction:
        train_set, val_set = train_test_split(train_set, test_size=val_fraction)

    return train_set, test_set, val_set


def get_target(y_true, y_pred):
    # Get skills and labels from y_true
    print(y_true)
    print(y_pred)
    mask = 1. - tf.cast(tf.equal(y_true, MASK_VALUE), y_true.dtype)
    y_true = y_true * mask
    skills, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)
    # Get predictions for each skill
    y_pred = tf.reduce_sum(y_pred * skills, axis=-1, keepdims=True)

    return y_true, y_pred
