import gc
from sklearn.model_selection import train_test_split

from Knowledge_Tracing.code.data_processing.load_preprocessed.load_preprocessed_data import \
    load_preprocessed_interactions
from Knowledge_Tracing.code.data_processing.preprocess.group_interactions_by_user_id import \
    generate_sequences_of_same_length
from Knowledge_Tracing.code.data_processing.load_preprocessed.saint_dataset import *


def get_saint_dataloaders(batch_size=128,
                          interactions_filepath="../input/assistmentds-2012/2012-2013-data-with-predictions-4-final.csv"
                          , output_filepath='/kaggle/working/', interaction_sequence_len=25, min_seq_len=5,
                          text_encoding_model=None, negative_value=0.0, mask_value=0.0, dictionary=None,
                          encoder_inputs=None, decoder_inputs=None, outputs=None):
    df = load_preprocessed_interactions(interactions_filepath=interactions_filepath, dictionary=dictionary)
    print(df)
    # grouping based on user_id to get the data supply
    nb_questions = df['question_id'].max()
    nb_skills = df['skill'].max()
    print(df.where(df['problem_id'] == 97595))
    print("Grouping users...")

    group = generate_sequences_of_same_length(df, seq_len=interaction_sequence_len, min_seq_len=min_seq_len,
                                              output_filepath=output_filepath)
    del df
    gc.collect()
    print(group)
    group = group[["user_id", "question_id", "problem_id", "correct", "elapsed_time", "skill"]]

    print("splitting")
    train, test = train_test_split(group, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print("train size: ", train.shape, "validation size: ", val.shape)
    complete_dict = ["question_id", "text_id", "skill",
                     "label", "r_elapsed_time", "text_encoding",
                     "input_question_id", "input_text_id", "input_skill",
                     "input_label", "input_r_elapsed_time", "input_text_encoding",
                     "target_id", "target_text_id", "target_skill",
                     "target_r_elapsed_time", 'target_label', "target_text_encoding"]
    if not encoder_inputs:
        encoder_inputs = complete_dict
    if not decoder_inputs:
        decoder_inputs = complete_dict
    if not outputs:
        outputs = complete_dict
    inputs_output_dict = {"encoder": encoder_inputs, "decoder": decoder_inputs, "output": outputs}
    train_dataset = SAINT_Dataset(train.values, text_encoding_model=text_encoding_model,
                                  max_seq=interaction_sequence_len, negative_value=negative_value,
                                  mask_value=mask_value,
                                  inputs_output_dict=inputs_output_dict)
    val_dataset = SAINT_Dataset(val.values, text_encoding_model=text_encoding_model,
                                max_seq=interaction_sequence_len, negative_value=negative_value, mask_value=mask_value,
                                inputs_output_dict=inputs_output_dict)
    test_dataset = SAINT_Dataset(test.values, text_encoding_model=text_encoding_model,
                                 max_seq=interaction_sequence_len, negative_value=negative_value, mask_value=mask_value,
                                 inputs_output_dict=inputs_output_dict)
    encoding_depth = train_dataset.encoding_depth
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=2,
                              shuffle=True)
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
    return train_loader, val_loader, test_loader, nb_questions, nb_skills, encoding_depth
