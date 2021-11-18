import gc

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from Knowledge_Tracing.code.data_processing.load_preprocessed.load_preprocessed_data import \
    load_preprocessed_interactions, load_preprocessed_texts
from Knowledge_Tracing.code.data_processing.preprocess.group_interactions_by_user_id import generate_sequences_of_same_length
from Knowledge_Tracing.code.models.encoding_models.BERTopic_model import BERTopic_model

MASK_VALUE = -1.0  # The masking value cannot be zero.


def load_dataset(batch_size=32, shuffle=True,
                 interactions_filepath="../input/assistmentds-2012/2012-2013-data-with-predictions-4-final"
                                       ".csv",
                 save_filepath='/kaggle/working/', texts_filepath='../input/',
                 interaction_sequence_len=30, min_df=2, max_df=1.0, max_features=1000):
    df = load_preprocessed_interactions(interactions_filepath=interactions_filepath)
    print(df)
    # grouping based on user_id to get the data supply
    nb_questions = len(df['question_id'].unique())
    nb_skills = len(df['skill'].unique())
    print("Grouping users...")

    group = generate_sequences_of_same_length(df, seq_len=interaction_sequence_len, output_filepath='/kaggle/working')
    del df
    gc.collect()
    print(group)
    group = group[["user_id", "question_id", "problem_id", "correct", "elapsed_time", "skill"]]

    print("splitting")
    train, test = train_test_split(group, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print("train size: ", train.shape, "validation size: ", val.shape)
    print(df)
    df = df[['user_id', 'problem_id', 'correct']]
    print(df)
    # Step 3.1 - Generate NLP extracted encoding for problems
    text_df = load_preprocessed_texts(texts_filepath=texts_filepath)
    encode_model = BERTopic_model()
    encode_model.fit(text_df, save_filepath)

    def generate_encodings():
        for name, group in df.groupby('user_id'):
            document_to_term = []
            labels = np.array([], dtype=np.int)
            for problem, label in list(zip(group['problem_id'].values, group['correct'].values)):
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
