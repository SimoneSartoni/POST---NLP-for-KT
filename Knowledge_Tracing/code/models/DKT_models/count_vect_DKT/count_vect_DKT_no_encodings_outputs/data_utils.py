import gc

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from Knowledge_Tracing.code.models.encoding_models.count_vectorizer import count_vectorizer
from Knowledge_Tracing.code.data_processing.load_preprocessed.load_preprocessed_data import load_preprocessed_texts, \
    load_preprocessed_interactions
from Knowledge_Tracing.code.data_processing.preprocess.group_interactions_by_user_id import generate_sequences_of_same_length

MASK_VALUE = -1.0  # The masking value cannot be zero.


def load_dataset(batch_size=32, shuffle=True,
                 interactions_filepath="../input/assistmentds-2012/2012-2013-data-with-predictions-4-final"
                                       ".csv",
                 save_filepath='/kaggle/working/', texts_filepath='../input/', min_df=2, max_df=1.0,
                 max_features=1000, interaction_sequence_len=30):
    df = load_preprocessed_interactions(interactions_filepath=interactions_filepath)
    group = generate_sequences_of_same_length(df, seq_len=interaction_sequence_len, output_filepath='/kaggle/working')
    del df
    gc.collect()
    text_df = load_preprocessed_texts(texts_filepath=texts_filepath)
    print(group)
    group = group[["user_id", "question_id", "problem_id", "correct", "elapsed_time", "skill"]]

    print("splitting")
    train, test = train_test_split(group, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print("train size: ", train.shape, "validation size: ", val.shape)

    # Step 3.1 - Generate NLP extracted encoding for problems
    encode_model = count_vectorizer(min_df=min_df, max_df=max_df, binary=False, max_features=max_features)
    encode_model.fit(text_df, save_filepath)

    max_value = encode_model.words_num
    del text_df
    gc.collect()
    print("number of words is: " + str(max_value))



    return train_set, val_set, test_set, encoding_depth


def get_target(y_true, y_pred, nb_encodings=300):
    mask = 1 - tf.cast(tf.equal(y_true, MASK_VALUE), y_true.dtype)
    y_true = y_true * mask
    y_pred = y_pred * mask
    return y_true, y_pred
