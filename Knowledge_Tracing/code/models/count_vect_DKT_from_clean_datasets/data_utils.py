import pandas as pd
import tensorflow as tf
import numpy as np
from Knowledge_Tracing.code.data_processing.dataset import dataset as dt
from Knowledge_Tracing.code.evaluation.predictors.logistic_regression import logistic_regressor
from code.models.count_vectorizer.count_vectorizer import count_vectorizer

MASK_VALUE = -1.0  # The masking value cannot be zero.


def load_dataset_NLP_skills(fn, batch_size=32, shuffle=True):

    # Step 3.1 - Generate NLP extracted encoding for problems
    loaded_dataset = dt(name="junyi", path="/Knowledge_Tracing/intermediate_files", prefix="clean_datasets/")
    loaded_dataset.load_interactions(standard_timestamps=False)
    loaded_dataset.load_saved_texts()
    loaded_dataset.compute_intersection_texts_and_interactions()
    encode_model = count_vectorizer(min_df=1, max_df=1.0, binary=False)
    encode_model.fit(loaded_dataset.interacted_with_text_problem_set, loaded_dataset.problem_id_to_index,
                     loaded_dataset.texts_list)

    # Step 3.2 - Remove problems without encoding (because we do not have text)
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

    nb_encodings = 100
    max_value = encode_model.words_num

    print("number of words is: " + str(max_value))

    def generate_encodings():
        dataset = loaded_dataset
        for problems, label_s, lengths in list(zip(dataset.problems, dataset.labels, dataset.lengths)):
            document_to_term = []
            labels = np.array([], dtype=np.int)
            for index in range(0, min(100, lengths)):
                encoding = encode_model.get_encoding(problems[index])
                encoding = np.expand_dims(encoding, axis=0)
                document_to_term.append(encoding)
                labels = np.append(labels, label_s[index])
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

    nb_users = len(loaded_dataset.problems)
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