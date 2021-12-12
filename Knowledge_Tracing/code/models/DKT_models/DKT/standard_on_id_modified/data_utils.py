import tensorflow as tf
from sklearn.model_selection import train_test_split

from Knowledge_Tracing.code.data_processing.load_preprocessed.get_DKT_dataloaders import get_DKT_dataloaders

MASK_VALUE = -1.  # The masking value cannot be zero.
from Knowledge_Tracing.code.data_processing.load_preprocessed.load_preprocessed_data import load_preprocessed_texts, \
    load_preprocessed_interactions
import sys


def create_dataset(generator, ids_depth, nb_questions, shuffle=True, batch_size=1024):
    input_types = {"feature_id": tf.float32}
    output_types = {"target_label": tf.float32, "target_id": tf.int32}

    input_shapes = {"feature_id": [None, ids_depth]}
    output_shapes = {"target_label": [None], "target_id": [None]}
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
            {"input_feature_id": inputs['feature_id']},
            tf.concat(
                values=[
                    tf.one_hot(outputs['target_id'], depth=nb_questions),
                    tf.expand_dims(outputs['target_label'], -1)
                ],
                axis=-1
            )
        )
    )

    print(dataset)

    # Step 7 - Pad sequences per batch
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=({"input_feature_id": [None, None]}, [None, None]),
        padding_values=({"input_feature_id": MASK_VALUE}, MASK_VALUE),
        drop_remainder=True
    )
    return dataset


def load_dataset(batch_size=32, shuffle=True,
                 interactions_filepath="../input/assistmentds-2012/2012-2013-data-with-predictions-4-final.csv",
                 save_filepath='/kaggle/working/', texts_filepath='../input/',
                 interaction_sequence_len=30, min_seq_len=5, dictionary=None):
    inputs = {"question_id": False, "text_id": False, "skill": False, "feature_id": True,
              "label": False, "r_elapsed_time": False, 'text_encoding': False, "target_id": False, "feature": False,
              "target_text_id": False, "target_skill": False, 'target_label': False, 'target_text_encoding': False,
              "target_feature": False, "target_feature_id": False}
    outputs = {"question_id": False, "text_id": False, "skill": False, "feature_id": False,
               "label": False, "r_elapsed_time": False, 'text_encoding': False, "target_id": True, "feature": False,
               "target_text_id": False, "target_skill": False, 'target_label': True, 'target_text_encoding': False,
               "target_feature": False, "target_feature_id": False, "target_feature_id": False}

    train_gen, val_gen, test_gen, \
    nb_questions, nb_skills = get_DKT_dataloaders(batch_size, shuffle, interactions_filepath,
                                                  output_filepath=save_filepath,
                                                  interaction_sequence_len=interaction_sequence_len
                                                  , min_seq_len=min_seq_len,
                                                  text_encoding_model=None,
                                                  negative_correctness=False,
                                                  inputs_dict=inputs, outputs_dict=outputs,
                                                  encode_correct_in_encodings=False,
                                                  encode_correct_in_skills=False,
                                                  encode_correct_in_id=True,
                                                  dictionary=dictionary)

    ids_depth = 2 * nb_questions
    train_loader = create_dataset(train_gen, ids_depth, nb_questions, shuffle=shuffle, batch_size=batch_size)
    val_loader = create_dataset(val_gen, ids_depth, nb_questions, shuffle=shuffle, batch_size=batch_size)
    test_loader = create_dataset(test_gen, ids_depth, nb_questions, shuffle=shuffle, batch_size=batch_size)

    return train_loader, val_loader, test_loader, ids_depth, nb_questions


def get_target(y_true, y_pred):
    # Get skills and labels from y_true

    # mask = 1. - tf.cast(tf.equal(y_true, MASK_VALUE), y_true.dtype)
    # y_true = y_true * mask
    skills, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)
    y_pred = tf.reduce_sum(y_pred * skills, axis=-1, keepdims=True)
    return y_true, y_pred
