import pandas as pd
import tensorflow as tf
import numpy as np
from Knowledge_Tracing.code.data_processing.dataset import dataset as dt
from Knowledge_Tracing.code.evaluation.predictors.logistic_regression import logistic_regressor
from code.models.gensim_model.gensim_pretrained_word2vec import pretrained_word2vec

MASK_VALUE = -1.  # The masking value cannot be zero.


def load_dataset_NLP_skills(fn, batch_size=32, shuffle=True):
    df = pd.read_csv(fn, encoding="ISO-8859-1")

    if "skill_id" not in df.columns:
        raise KeyError(f"The column 'skill_id' was not found on {fn}")
    if "correct" not in df.columns:
        raise KeyError(f"The column 'correct' was not found on {fn}")
    if "user_id" not in df.columns:
        raise KeyError(f"The column 'user_id' was not found on {fn}")

    if not (df['correct'].isin([0, 1])).all():
        raise KeyError(f"The values of the column 'correct' must be 0 or 1.")

    # Step 1.1 - Remove questions without skill
    df.dropna(subset=['skill_id'], inplace=True)

    # Step 1.2 - Remove users with a single answer
    df = df.groupby('user_id').filter(lambda q: len(q) > 1).copy()

    # Step 2 - Enumerate skill id
    df['skill'], _ = pd.factorize(df['skill_id'], sort=True)
    print(df['skill'])

    df['skill_with_answer'] = df['skill'] * 2 + df['correct']

    # Step 3.1 - Generate NLP extracted encoding for problems
    loaded_dataset = dt(name="assistments_2009", path="/Knowledge_Tracing/intermediate_files", prefix="clean_datasets/")
    loaded_dataset.load_saved_texts()
    encode_model = pretrained_word2vec(load=True)
    encode_model.fit()
    encode_model.encode_problems(loaded_dataset.problem_id_to_index, loaded_dataset.texts_list)

    # Step 3.2 - Remove problems without encoding (because we do not have text)
    df = df.loc[df['problem_id'].isin(loaded_dataset.problem_id_to_index)]
    print("start_nlp")
    nlp_encodings = [tf.constant(encode_model.get_encoding(problem), shape=encode_model.vector_size,
                                 dtype=tf.float32) for problem in df['problem_id']]
    print("start_wrong")
    df['encodings_wrong'] = [encoding if correct == 0 else
                             tf.constant(0.0, shape=encode_model.vector_size, dtype=tf.float32)
                             for encoding, correct in list(zip(nlp_encodings, df['correct']))]
    print("start_correct")
    df['encodings_correct'] = [encoding if correct else
                               tf.constant(0.0, shape=encode_model.vector_size, dtype=tf.float32)
                               for encoding, correct in list(zip(nlp_encodings, df['correct']))]
    print(df['encodings_correct'].iloc[0])

    encoding_depth = encode_model.vector_size
    skill_depth = df['skill'].max() + 1
    features_depth = df['skill_with_answer'].max() + 1
    # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    seq = df.groupby('user_id').apply(
        lambda r: (
             r['encodings_correct'].to_numpy()[:-1],
             r['encodings_wrong'].to_numpy()[:-1],
             tf.one_hot(r['skill_with_answer'].to_numpy()[:-1], depth=features_depth),
             r['encodings_correct'].to_numpy()[1:],
             r['encodings_wrong'].to_numpy()[1:],
             tf.one_hot(r['skill_with_answer'].to_numpy()[1:], depth=features_depth),
             r['correct'].to_numpy()[1:]
        )
    )
    nb_users = len(seq)

    # Step 5 - Get Tensorflow Dataset
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: seq,
        output_types=(tf.float32, tf.float32, tf.int32, tf.float32, tf.float32, tf.int32, tf.int32),
        output_shapes=([None, encode_model.vector_size], [None, encode_model.vector_size], [None, features_depth],
                       [None, encode_model.vector_size], [None, encode_model.vector_size], [None, features_depth],
                       [None, 1])
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=nb_users)

    print(dataset)
    dataset = dataset.map(
        lambda i_enc_corr, i_enc_wrong, i_feat, o_enc_corr, o_enc_wrong, o_feat, o_label: (
            {'input_encoding_correct': i_enc_corr,
             'input_encoding_wrong': i_enc_wrong,
             'input_feature': tf.cast(i_feat, dtype=tf.float32, name="cast_to_float")},
            tf.concat(values=[
                o_enc_corr,
                o_enc_wrong,
                tf.cast(o_feat, dtype=tf.float32, name="cast_to_float"),
                tf.cast(o_label, dtype=tf.float32, name="cast_to_float")],
                axis=-1)
        )
    )

    # Step 6 - Encode categorical features and merge skills with labels to compute target loss.
    # More info: https://github.com/tensorflow/tensorflow/issues/32142

    print(dataset)

    # Step 7 - Pad sequences per batch
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padding_values=(-1.0),
        drop_remainder=True
    )
    """padded_shapes=({'input_correct': [None], 'input_skill': [None, None],
                            'input_encoding': [None, encoding_depth]},
                           {'output_correct': [None], 'output_skill': [None, None],
                            'output_encoding': [None, encoding_depth]}),"""
    print(dataset)

    length = nb_users // batch_size
    return dataset, length, features_depth, encoding_depth


def get_target(y_true, y_pred, nb_encodings=300, nb_features=286):
    # Get skills and labels from y_true
    print(y_true)
    print(y_pred)

    corrects_true, wrong_true, features_true, label = tf.split(y_true, num_or_size_splits=[nb_encodings, nb_encodings,
                                                                                           nb_features, 1], axis=-1)
    corrects_pred, wrong_pred, features_pred = tf.split(y_pred, num_or_size_splits=[nb_encodings, nb_encodings,
                                                                                    nb_features], axis=-1)

    corrects_mask = 1. - tf.cast(tf.equal(corrects_true, -1.0), corrects_true.dtype)
    wrong_mask = 1. - tf.cast(tf.equal(wrong_true, -1.0), wrong_true.dtype)
    features_mask = 1. - tf.cast(tf.equal(features_true, -1.0), features_true.dtype)

    corrects_true = corrects_true * corrects_mask
    wrong_true = wrong_true * wrong_mask
    features_true = features_true * features_mask
    # Get predictions for each skill
    corrects_pred = tf.reduce_sum(corrects_pred * corrects_true, axis=-1, keepdims=True)
    wrong_pred = tf.reduce_sum(wrong_pred * wrong_true, axis=-1, keepdims=True)
    features_pred = tf.reduce_sum(features_pred * features_true, axis=-1, keepdims=True)
    y_pred = corrects_pred + wrong_pred + features_pred
    y_true = label
    return y_true, y_pred
