import gc

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from Knowledge_Tracing.code.models.encoding_models.count_vectorizer import count_vectorizer
from Knowledge_Tracing.code.data_processing.load_preprocessed.load_preprocessed_data import load_preprocessed_texts, \
    load_preprocessed_interactions
from Knowledge_Tracing.code.data_processing.load_preprocessed.get_DKT_dataloaders import get_DKT_dataloaders

MASK_VALUE = -1.0  # The masking value cannot be zero.


def load_dataset(batch_size=32, shuffle=True,
                 interactions_filepath="../input/assistmentds-2012/2012-2013-data-with-predictions-4-final.csv",
                 save_filepath='/kaggle/working/', texts_filepath='../input/', min_df=2, max_df=1.0,
                 max_features=1000, interaction_sequence_len=25, min_seq_len=5, encode_correct_in_encodings=False):
    inputs = {"question_id": False, "text_id": False, "skill": False,
              "label": False, "r_elapsed_time": False, 'text_encoding': True, "target_id": False,
              "target_text_id": False, "target_skill": False, 'target_label': False, 'target_text_encoding': True}
    outputs = {"question_id": False, "text_id": False, "skill": False,
               "label": False, "r_elapsed_time": False, "target_id": False,
               "target_text_id": False, "target_skill": False, 'target_label': True}

    text_df = load_preprocessed_texts(texts_filepath=texts_filepath)
    # Step 3.1 - Generate NLP extracted encoding for problems
    encode_model = count_vectorizer(min_df=min_df, max_df=max_df, binary=False, max_features=max_features)
    encode_model.fit(text_df, save_filepath)

    train_loader, val_loader, test_loader, nb_questions = get_DKT_dataloaders(batch_size, shuffle, interactions_filepath,
                                                                              output_filepath='/kaggle/working/',
                                                                              interaction_sequence_len=interaction_sequence_len
                                                                              , min_seq_len=min_seq_len,
                                                                              text_encoding_model=encode_model,
                                                                              negative_correctness=False,
                                                                              inputs=inputs, outputs=outputs,
                                                                              encode_correct_in_encodings=
                                                                              encode_correct_in_encodings)
    encoding_depth = 2 * encode_model.vector_size

    return train_loader, val_loader, test_loader, encoding_depth


def get_target(y_true, y_pred, nb_encodings=300):
    mask = 1 - tf.cast(tf.equal(y_true, MASK_VALUE), y_true.dtype)
    y_true = y_true * mask
    y_pred = y_pred * mask
    encodings_true, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)
    encodings_pred, y_pred = tf.split(y_pred, num_or_size_splits=[-1, 1], axis=-1)

    return y_true, y_pred
