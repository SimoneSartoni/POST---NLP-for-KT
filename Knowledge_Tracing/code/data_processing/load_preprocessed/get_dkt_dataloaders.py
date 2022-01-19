import gc
from sklearn.model_selection import train_test_split
from Knowledge_Tracing.code.data_processing.load_preprocessed.load_preprocessed_data import \
    load_preprocessed_interactions
from Knowledge_Tracing.code.data_processing.preprocess.group_interactions_by_user_id import \
    generate_sequences_of_same_length
from Knowledge_Tracing.code.data_processing.load_preprocessed.dkt_dataset import *


def get_DKT_dataloaders(batch_size=128, shuffle=False,
                        interactions_filepath="../input/assistmentds-2012/2012-2013-data-with-predictions-4-final."
                        "csv", output_filepath='/content/', interaction_sequence_len=25, min_seq_len=5,
                        text_encoding_model=None, negative_correctness=False, inputs_dict={}, outputs_dict={},
                        encode_correct_in_encodings=False, encode_correct_in_skills=False, encode_correct_in_id=False,
                        dictionary=None):
    df = load_preprocessed_interactions(interactions_filepath=interactions_filepath, dictionary=dictionary)
    print(df)
    # grouping based on user_id to get the data supply
    nb_questions = len(df['question_id'].max()) + 1
    nb_skills = len(df['skill'].max()) + 1
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

    train_dataset = DKT_Dataset(train.values, text_encoding_model=text_encoding_model, max_seq=interaction_sequence_len,
                                negative_correctness=negative_correctness, inputs_dict=inputs_dict, outputs_dict=outputs_dict,
                                encode_correct_in_encodings=encode_correct_in_encodings,
                                encode_correct_in_id=encode_correct_in_id,
                                encode_correct_in_skills=encode_correct_in_skills, nb_skills=nb_skills,
                                nb_questions=nb_questions)
    val_dataset = DKT_Dataset(val.values, text_encoding_model=text_encoding_model, max_seq=interaction_sequence_len,
                              negative_correctness=negative_correctness, inputs_dict=inputs_dict, outputs_dict=outputs_dict,
                              encode_correct_in_encodings=encode_correct_in_encodings,
                              encode_correct_in_id=encode_correct_in_id,
                              encode_correct_in_skills=encode_correct_in_skills, nb_skills=nb_skills,
                              nb_questions=nb_questions)
    test_dataset = DKT_Dataset(test.values, text_encoding_model=text_encoding_model, max_seq=interaction_sequence_len,
                               negative_correctness=negative_correctness, inputs_dict=inputs_dict, outputs_dict=outputs_dict,
                               encode_correct_in_encodings=encode_correct_in_encodings,
                               encode_correct_in_id=encode_correct_in_id,
                               encode_correct_in_skills=encode_correct_in_skills, nb_skills=nb_skills,
                               nb_questions=nb_questions)

    return train_dataset, val_dataset, test_dataset, nb_questions, nb_skills
