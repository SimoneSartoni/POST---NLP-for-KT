import pandas as pd
import tensorflow as tf
import numpy as np
from Knowledge_Tracing.code.data_processing.dataset import dataset as dt
from Knowledge_Tracing.code.models.count_vectorizer.count_vectorizer import count_vectorizer

MASK_VALUE = -1.  # The masking value cannot be zero.


def generate_encodings(df, encode_model):
    inputs_generator = []
    outputs_generator = []
    df_users = df.groupby('user_id')
    for name in df_users.groups.keys():
        r = df_users.get_group(name)
        document_to_term = []
        labels = np.array([], dtype=np.float)
        for problem, label in list(zip(r['problem_id'], r['correct'])):
            encoding = np.asarray(encode_model.get_encoding(problem)).astype('float32')
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
        inputs_generator.append(inputs)
        outputs_generator.append(outputs)
    return inputs_generator, outputs_generator


def load_dataset_NLP_skills(fn, batch_size=32, shuffle=True, repository="", keyedvectors=""):
    df = pd.read_csv(fn, encoding="ISO-8859-1")

    if "skill_id" not in df.columns:
        raise KeyError(f"The column 'skill_id' was not found on {fn}")
    if "correct" not in df.columns:
        raise KeyError(f"The column 'correct' was not found on {fn}")
    if "user_id" not in df.columns:
        raise KeyError(f"The column 'user_id' was not found on {fn}")

    if not (df['correct'].isin([0, 1])).all():
        raise KeyError(f"The values of the column 'correct' must be 0 or 1.")

    # Step 1.2 - Remove users with a single answer
    df = df.groupby('user_id').filter(lambda q: len(q) > 1).copy()

    # Step 3.1 - Generate NLP extracted encoding for problems
    loaded_dataset = dt(name="assistments_2012", path=repository, prefix="clean_datasets/")
    loaded_dataset.load_interactions(standard_timestamps=False)
    loaded_dataset.load_saved_texts()
    loaded_dataset.compute_intersection_texts_and_interactions()
    encode_model = count_vectorizer(min_df=1, max_df=1.0, binary=False)
    encode_model.fit(loaded_dataset.interacted_with_text_problem_set, loaded_dataset.problem_id_to_index,
                     loaded_dataset.texts_list)

    # Step 3.2 - Remove problems without encoding (because we do not have text)
    df = df.loc[df['problem_id'].isin(encode_model.problem_id_to_index)]
    """print("start_nlp")
    nlp_encodings = [np.asarray(encode_model.get_encoding(problem)).astype('float32') for problem in df['problem_id']]

    print("start_wrong")
    df['encodings_wrong'] = [encoding if correct == 0 else
                             np.zeros(shape=encode_model.vector_size, dtype=np.float)
                             for encoding, correct in list(zip(nlp_encodings, df['correct']))]
    print("start_correct")
    df['encodings_correct'] = [encoding if correct else
                               np.zeros(shape=encode_model.vector_size, dtype=np.float)
                               for encoding, correct in list(zip(nlp_encodings, df['correct']))]"""

    inputs, outputs = generate_encodings(df, encode_model)

    def generator():
        for i, o in list(zip(inputs, outputs)):
            yield i, o

    encoding_depth = encode_model.vector_size

    types = ((tf.float32, tf.float32),
             (tf.float32, tf.float32))
    shapes = (([None, encode_model.vector_size], [None]),
              ([None, encode_model.vector_size], [None]))
    # Step 5 - Get Tensorflow Dataset
    dataset = tf.data.Dataset.from_generator(
        generator=generator,
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


"""corrects_mask = 1. - tf.cast(tf.equal(corrects_true, -1.0), corrects_true.dtype)
    wrong_mask = 1. - tf.cast(tf.equal(wrong_true, -1.0), wrong_true.dtype)
    features_mask = 1. - tf.cast(tf.equal(features_true, -1.0), features_true.dtype)

    corrects_true = corrects_true * corrects_mask
    wrong_true = wrong_true * wrong_mask
    features_true = features_true * features_mask
    # Get predictions for each skill
    corrects_pred = tf.reduce_sum(corrects_pred * corrects_true, axis=-1, keepdims=True)
    wrong_pred = tf.reduce_sum(wrong_pred * wrong_true, axis=-1, keepdims=True)
    features_pred = tf.reduce_sum(features_pred * features_true, axis=-1, keepdims=True)"""