import Knowledge_Tracing.code.models.complex_models.config as config

import numpy as np
from torch.utils.data import Dataset, DataLoader
import gc
from sklearn.model_selection import train_test_split

from Knowledge_Tracing.code.data_processing.load_preprocessed.load_preprocessed_data import load_preprocessed_texts, \
    load_preprocessed_interactions
from Knowledge_Tracing.code.data_processing.preprocess.group_interactions_by_user_id import generate_sequences_of_same_length


class DKTDataset(Dataset):
    def __init__(self, grouped_df, text_encoding_model=None, max_seq=100, negative_correctness=False):
        self.max_seq = max_seq
        self.data = grouped_df
        self.negative_correctness = negative_correctness
        self.text_encoding_model = text_encoding_model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id, unique_question_id, text_id, answered_correctly, response_elapsed_time, exe_skill = self.data[idx]
        seq_len = len(unique_question_id)

        q_ids = np.zeros(self.max_seq, dtype=int)
        text_ids = np.zeros(self.max_seq, dtype=int)
        ans = np.zeros(self.max_seq, dtype=int)
        r_elapsed_time = np.zeros(self.max_seq, dtype=int)
        skill = np.zeros(self.max_seq, dtype=int)
        input_label = np.zeros(self.max_seq, dtype=int)
        input_r_elapsed_time = np.zeros(self.max_seq, dtype=int)

        q_ids[-seq_len:] = unique_question_id
        text_ids[-seq_len:] = text_id
        if self.negative_correctness:
            ans[-seq_len:] = [1.0 if x == 1.0 else -1.0 for x in answered_correctly]
        else:
            ans[-seq_len:] = answered_correctly
        r_elapsed_time[-seq_len:] = response_elapsed_time
        skill[-seq_len:] = exe_skill

        input_ids = q_ids
        input_text_ids = text_ids
        if self.text_encoding_model:
            input_text_encodings = [self.text_encoding_model.get_encoding(text_id) for text_id in text_ids]
        input_r_elapsed_time[1:] = r_elapsed_time[:-1].copy().astype(np.int)
        input_skill = skill
        input_label[1:] = ans[:-1]

        target_ids = input_ids[:]
        target_text_ids = input_text_ids
        target_skill = input_skill[:]
        target_label = ans

        encoder_inputs = {"question_id": input_ids, "text_id": input_text_ids, "skill": input_skill}
        if self.text_encoding_model:
            encoder_inputs["text_encoding"]=input_text_encodings
        decoder_inputs = {"label": input_label, "r_elapsed_time": input_r_elapsed_time}
        decoder_targets = {"target_id": target_ids, "target_text_id": target_text_ids, "target_skill": target_skill,
                           'target_label': target_label}
        inputs = {'encoder': encoder_inputs, 'decoder': decoder_inputs}
        return inputs, decoder_targets


def get_dataloaders(interactions_filepath="../input/assistmentds-2012/2012-2013-data-with-predictions-4-final"
                                       ".csv", texts_filepath='../input/',  output_filepath='/kaggle/working/',
                    interaction_sequence_len=25, personal_cleaning=True, text_encoding_model=None, negative_correctness=True):

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

    train_dataset = DKTDataset(train.values, text_encoding_model=text_encoding_model, max_seq=interaction_sequence_len,
                               negative_correctness=negative_correctness)
    val_dataset = DKTDataset(val.values, text_encoding_model=text_encoding_model, max_seq=interaction_sequence_len,
                             negative_correctness=negative_correctness)
    test_dataset = DKTDataset(test.values, text_encoding_model=text_encoding_model, max_seq=interaction_sequence_len,
                              negative_correctness=negative_correctness)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              num_workers=2,
                              shuffle=True)
    del train_dataset
    gc.collect()
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            num_workers=2,
                            shuffle=False)
    del val_dataset
    gc.collect()
    test_loader = DataLoader(test_dataset,
                             batch_size=config.BATCH_SIZE,
                             num_workers=2,
                             shuffle=False)
    del test_dataset
    gc.collect()
    return train_loader, val_loader, test_loader, nb_questions, nb_skills
