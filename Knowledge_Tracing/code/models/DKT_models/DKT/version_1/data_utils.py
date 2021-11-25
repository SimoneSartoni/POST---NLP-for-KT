import gc

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from Knowledge_Tracing.code.models.encoding_models.count_vectorizer import count_vectorizer
from Knowledge_Tracing.code.data_processing.load_preprocessed.load_preprocessed_data import load_preprocessed_texts, \
    load_preprocessed_interactions
from Knowledge_Tracing.code.data_processing.load_preprocessed.get_DKT_dataloaders import get_DKT_dataloaders

MASK_VALUE = -1.0  # The masking value cannot be zero.


"""
    possible_input_types = {"question_id": tf.float32, "text_id": tf.float32, "skill": tf.float32,
                            "label": tf.float32, "r_elapsed_time": tf.float32, "target_id": tf.float32,
                            "target_text_id": tf.float32, "target_skill": tf.float32,
                            'target_label': tf.float32}
    possible_input_shapes = {"question_id": [None], "text_id": [None, 1], "skill": [None],
                             "label": [None], "r_elapsed_time": [None], "target_id": [None],
                             "target_text_id": [None], "target_skill": [None],
                             'target_label': [None]}
"""


def create_dataset(generator, features_depth, shuffle=True, batch_size=1024):
    input_types = {"feature": tf.float32, "target_feature": tf.float32}
    output_types = {"target_label": tf.float32}

    input_shapes = {"feature": [None, features_depth], "target_feature": [None, features_depth]}
    output_shapes = {"target_label": [None]}
    types = (input_types, output_types)
    shapes = (input_shapes, output_shapes)
    dataset = tf.data.Dataset.from_generator(
        generator=generator.generator,
        output_types=types,
        output_shapes=shapes
    )

    nb_users = generator.__len__()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=nb_users, reshuffle_each_iteration=True)

    print(dataset)
    dataset = dataset.map(
        lambda inputs, outputs: (
            {"input_feature": inputs['feature'],
             "target_feature": inputs['target_feature']},
            tf.expand_dims(outputs['target_label'], axis=-1)
        )
    )

    print(dataset)

    # Step 7 - Pad sequences per batch
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padding_values=-1.0,
        drop_remainder=True
    )
    return dataset


def load_dataset(batch_size=32, shuffle=True,
                 interactions_filepath="../input/assistmentds-2012/2012-2013-data-with-predictions-4-final.csv",
                 save_filepath='/kaggle/working/', texts_filepath='../input/',
                 interaction_sequence_len=30, min_seq_len=5, dictionary=None):
    inputs = {"question_id": False, "text_id": False, "skill": False,
              "label": False, "r_elapsed_time": False, 'text_encoding': False, "target_id": False, "features": True,
              "target_text_id": False, "target_skill": False, 'target_label': False, 'target_text_encoding': False,
              "target_features": True}
    outputs = {"question_id": False, "text_id": False, "skill": False,
               "label": False, "r_elapsed_time": False, 'text_encoding': False, "target_id": False, "features": False,
               "target_text_id": False, "target_skill": False, 'target_label': True, 'target_text_encoding': False,
               "target_features": False}

    text_df = load_preprocessed_texts(texts_filepath=texts_filepath)
    # Step 3.1 - Generate NLP extracted encoding for problems

    train_gen, val_gen, test_gen, \
        nb_questions, nb_skills = get_DKT_dataloaders(batch_size, shuffle, interactions_filepath,
                                                      output_filepath='/kaggle/working/',
                                                      interaction_sequence_len=interaction_sequence_len
                                                      , min_seq_len=min_seq_len,
                                                      text_encoding_model=None,
                                                      negative_correctness=False,
                                                      inputs_dict=inputs, outputs_dict=outputs,
                                                      encode_correct_in_encodings=False,
                                                      encode_correct_in_skills=True,
                                                      dictionary=dictionary)
    features_depth = 2*nb_skills
    train_loader = create_dataset(train_gen, features_depth, shuffle=shuffle, batch_size=batch_size)
    val_loader = create_dataset(val_gen, features_depth, shuffle=shuffle, batch_size=batch_size)
    test_loader = create_dataset(test_gen, features_depth, shuffle=shuffle, batch_size=batch_size)

    return train_loader, val_loader, test_loader, features_depth


def get_target(y_true, y_pred, nb_encodings=300):
    mask = 1 - tf.cast(tf.equal(y_true, MASK_VALUE), y_true.dtype)
    y_true = y_true * mask
    y_pred = y_pred * mask
    return y_true, y_pred
