import gc
from sklearn.model_selection import train_test_split

from Knowledge_Tracing.code.data_processing.load_preprocessed.load_preprocessed_data import \
    load_preprocessed_interactions
from Knowledge_Tracing.code.data_processing.preprocess.group_interactions_by_user_id import generate_sequences_of_same_length
from Knowledge_Tracing.code.data_processing.load_preprocessed.DKT_dataset import *


def get_DKT_dataloaders(batch_size=128, shuffle=False, interactions_filepath="../input/assistmentds-2012/2012-2013-data-with-predictions-4-final."
                                                "csv", output_filepath='/kaggle/working/', interaction_sequence_len=25,
                          text_encoding_model=None, negative_correctness=False, **encodings_kwargs):


    df = load_preprocessed_interactions(interactions_filepath=interactions_filepath)
    print(df)
    # grouping based on user_id to get the data supply
    nb_questions = len(df['question_id'].unique())
    nb_skills = len(df['skill'].unique())
    print("Grouping users...")

    group = generate_sequences_of_same_length(df, seq_len=interaction_sequence_len, output_filepath=output_filepath)
    del df
    gc.collect()
    print(group)
    group = group[["user_id", "question_id", "problem_id", "correct", "elapsed_time", "skill"]]

    print("splitting")
    train, test = train_test_split(group, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print("train size: ", train.shape, "validation size: ", val.shape)

    train_dataset = DKT_Dataset(train.values, text_encoding_model=text_encoding_model, max_seq=interaction_sequence_len,
                               negative_correctness=negative_correctness, **encodings_kwargs)
    val_dataset = DKT_Dataset(val.values, text_encoding_model=text_encoding_model, max_seq=interaction_sequence_len,
                             negative_correctness=negative_correctness, **encodings_kwargs)
    test_dataset = DKT_Dataset(test.values, text_encoding_model=text_encoding_model, max_seq=interaction_sequence_len,
                              negative_correctness=negative_correctness, **encodings_kwargs)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=2,
                              shuffle=shuffle)
    del train_dataset
    gc.collect()
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            num_workers=2,
                            shuffle=False)
    del val_dataset
    gc.collect()
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             num_workers=2,
                             shuffle=False)
    del test_dataset
    gc.collect()
    return train_loader, val_loader, test_loader, nb_questions, nb_skills