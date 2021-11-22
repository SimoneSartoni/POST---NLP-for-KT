import gc

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from Knowledge_Tracing.code.models.encoding_models.count_vectorizer import count_vectorizer
from Knowledge_Tracing.code.models.encoding_models.count_vectorizer import count_vectorizer
from Knowledge_Tracing.code.data_processing.load_preprocessed.load_preprocessed_data import load_preprocessed_texts, \
    load_preprocessed_interactions
from Knowledge_Tracing.code.data_processing.preprocess.group_interactions_by_user_id import generate_sequences_of_same_length

MASK_VALUE = -1.0  # The masking value cannot be zero.


def load_dataset(batch_size=32, shuffle=True, dataset_name='assistment_2012',
                 interactions_filepath="../input/assistmentds-2012/2012-2013-data-with-predictions-4-final"
                                       ".csv",
                 save_filepath='/kaggle/working/', texts_filepath='../input/', min_df=2, max_df=1.0,
                 max_features=1000, interaction_sequence_len=25, min_seq_len=5):
    df = load_preprocessed_interactions(interactions_filepath=interactions_filepath)
    group = generate_sequences_of_same_length(df, seq_len=interaction_sequence_len, min_seq_len=min_seq_len, output_filepath='/kaggle/working')
    del df
    gc.collect()
    text_df = load_preprocessed_texts(texts_filepath=texts_filepath)
    print(group)
    group = group[["user_id", "question_id", "problem_id", "correct", "elapsed_time", "skill"]]
    # Step 3.1 - Generate NLP extracted encoding for problems
    encode_model = count_vectorizer(min_df=min_df, max_df=max_df, binary=False, max_features=max_features)
    encode_model.fit(text_df, save_filepath)

    max_value = encode_model.words_num
    del text_df
    gc.collect()
    print("number of words is: " + str(max_value))
    print("splitting")
    train, test = train_test_split(group, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print("train size: ", train.shape, "validation size: ", val.shape)

    def generate_encodings_val():
        for group in val.values:
            document_to_term = []
            labels = np.array([], dtype=np.int)
            user_ids, unique_question_ids, text_ids, corrects, response_elapsed_times, exe_skills = group
            for text_id, label in list(zip(text_ids, corrects)):
                encoding = encode_model.get_encoding(problem=text_id)
                encoding = np.expand_dims(encoding, axis=0)
                document_to_term.append(encoding)
                labels = np.append(labels, label)
            document_to_term = np.concatenate(document_to_term, axis=0)
            i_doc = document_to_term[:-1]
            o_doc = document_to_term[1:]
            i_label = labels[:-1]
            o_label = labels[1:]
            inputs = (i_doc, i_label, o_doc)
            outputs = o_label
            yield inputs, outputs

    def generate_encodings_test():
        for group in test.values:
            document_to_term = []
            labels = np.array([], dtype=np.int)
            user_ids, unique_question_ids, text_ids, corrects, response_elapsed_times, exe_skills = group
            for text_id, label in list(zip(text_ids, corrects)):
                encoding = encode_model.get_encoding(problem=text_id)
                encoding = np.expand_dims(encoding, axis=0)
                document_to_term.append(encoding)
                labels = np.append(labels, label)
            document_to_term = np.concatenate(document_to_term, axis=0)
            i_doc = document_to_term[:-1]
            o_doc = document_to_term[1:]
            i_label = labels[:-1]
            o_label = labels[1:]
            inputs = (i_doc, i_label, o_doc)
            outputs = o_label
            yield inputs, outputs

    def generate_encodings_train():
        for group in train.values:
            document_to_term = []
            labels = np.array([], dtype=np.int)
            user_ids, unique_question_ids, text_ids, corrects, response_elapsed_times, exe_skills = group
            for text_id, label in list(zip(text_ids, corrects)):
                encoding = encode_model.get_encoding(problem=text_id)
                encoding = np.expand_dims(encoding, axis=0)
                document_to_term.append(encoding)
                labels = np.append(labels, label)
            document_to_term = np.concatenate(document_to_term, axis=0)
            i_doc = document_to_term[:-1]
            o_doc = document_to_term[1:]
            i_label = labels[:-1]
            o_label = labels[1:]
            inputs = (i_doc, i_label, o_doc)
            outputs = o_label
            yield inputs, outputs

    encoding_depth = encode_model.vector_size

    def create_dataset(generate_encodings, input_dataset, encoding_depth):
        # Step 5 - Get Tensorflow Dataset
        types = ((tf.float32, tf.float32, tf.float32),
                 tf.float32)
        shapes = (([None, encoding_depth], [None], [None, encoding_depth]),
                  [None])
        # Step 5 - Get Tensorflow Dataset
        dataset = tf.data.Dataset.from_generator(
            generator=generate_encodings,
            output_types=types,
            output_shapes=shapes
        )

        nb_users = len(input_dataset.values)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=nb_users, reshuffle_each_iteration=True)

        print(dataset)
        dataset = dataset.map(
            lambda inputs, outputs: (
                (inputs[0], tf.expand_dims(inputs[1], axis=-1), inputs[2]),
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

    train_set = create_dataset(generate_encodings_train, train, encoding_depth)
    val_set = create_dataset(generate_encodings_val, val, encoding_depth)
    test_set = create_dataset(generate_encodings_test, test, encoding_depth)

    return train_set, val_set, test_set, encoding_depth


def get_target(y_true, y_pred, nb_encodings=300):
    mask = 1 - tf.cast(tf.equal(y_true, MASK_VALUE), y_true.dtype)
    y_true = y_true * mask
    y_pred = y_pred * mask
    return y_true, y_pred
