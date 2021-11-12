import Knowledge_Tracing.code.models.complex_models.config as config
from Knowledge_Tracing.code.utils.utils import try_parsing_date
from datetime import datetime

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gc
from sklearn.model_selection import train_test_split

from Knowledge_Tracing.code.data_processing.get_data_assistments_2009 import get_data_assistments_2009
from Knowledge_Tracing.code.data_processing.get_data_assistments_2012 import get_data_assistments_2012


class DKTDataset(Dataset):
    def __init__(self, group, max_seq=100):
        self.samples = group
        self.max_seq = max_seq
        self.data = []

        for unique_question_id, text_id, ans, res_time, exe_skill in self.samples:
            if len(unique_question_id) >= self.max_seq:
                self.data.extend([(unique_question_id[l:l + self.max_seq], text_id[l:l + self.max_seq],
                                   ans[l:l + self.max_seq], res_time[l:l + self.max_seq], exe_skill[l:l + self.max_seq])
                                  for l in range(len(unique_question_id)) if l % self.max_seq == 0])
            elif self.max_seq > len(unique_question_id) > 1:
                self.data.append((unique_question_id, text_id, ans, res_time, exe_skill))
            else:
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        unique_question_id, text_id, answered_correctly, response_elapsed_time, exe_skill = self.data[idx]
        seq_len = len(unique_question_id)

        q_ids = np.zeros(self.max_seq, dtype=int)
        text_ids = np.zeros(self.max_seq, dtype=int)
        ans = np.zeros(self.max_seq, dtype=int)
        r_elapsed_time = np.zeros(self.max_seq, dtype=int)
        skill = np.zeros(self.max_seq, dtype=int)
        input_label = np.zeros(self.max_seq, dtype=int)

        if seq_len >= self.max_seq:
            q_ids[:] = unique_question_id[-self.max_seq:]
            text_ids[:] = text_id[-self.max_seq]
            ans[:] = answered_correctly[-self.max_seq:]
            r_elapsed_time[:] = response_elapsed_time[-self.max_seq:]
            skill[:] = exe_skill[-self.max_seq:]
        else:
            q_ids[-seq_len:] = unique_question_id
            text_ids[-seq_len:] = text_id
            ans[-seq_len:] = answered_correctly
            r_elapsed_time[-seq_len:] = response_elapsed_time
            skill[-seq_len:] = exe_skill

        input_ids = q_ids
        input_text_ids = text_ids
        input_r_elapsed_time = r_elapsed_time[:-1].copy().astype(np.int)
        input_skill = skill
        input_label[1:] = ans[:-1]

        target_ids = input_ids[:]
        target_text_ids = input_text_ids
        target_skill = input_skill[:]
        target_label = ans

        encoder_inputs = {"question_id": input_ids, "text_id": input_text_ids, "skill": input_skill}
        decoder_inputs = {"label": input_label, "r_elapsed_time": input_r_elapsed_time}
        decoder_targets = {"target_id": target_ids, "target_text_id": target_text_ids, "target_skill": target_skill, 'target_label': target_label}
        inputs = {}
        inputs['encoder'] = encoder_inputs
        inputs['decoder'] = decoder_inputs

        return inputs, decoder_targets


def get_dataloaders(batch_size=32, shuffle=True, dataset_name='assistment_2012',
                 interactions_filepath="../input/assistmentds-2012/2012-2013-data-with-predictions-4-final"
                                       ".csv",
                 save_filepath='/kaggle/working/', texts_filepath='../input/', min_df=2, max_df=1.0,
                 min_questions=2, max_features=1000, max_questions=25, n_rows=None, n_texts=None,
                 personal_cleaning=True):
    if dataset_name == 'assistment_2012':
        df, text_df = get_data_assistments_2012(min_questions=min_questions, max_questions=max_questions,
                                                interactions_filepath=interactions_filepath,
                                                texts_filepath=texts_filepath, n_rows=n_rows, n_texts=n_texts,
                                                make_sentences_flag=False, personal_cleaning=personal_cleaning)
    elif dataset_name == 'assistment_2009':
        df, text_df = get_data_assistments_2009(min_questions=min_questions, max_questions=max_questions,
                                                interactions_filepath=interactions_filepath,
                                                texts_filepath=texts_filepath, n_rows=n_rows, n_texts=n_texts,
                                                make_sentences_flag=False, personal_cleaning=personal_cleaning, )
    del text_df
    gc.collect()
    print(df)
    df = df[["user_id", "problem_id", "question_id", "correct", "elapsed_time", "skill"]]

    # grouping based on user_id to get the data supply
    print("Grouping users...")
    nb_questions = len(df['question_id'].unique())
    nb_skills = len(df['skill'].unique())

    group = df.groupby("user_id").apply(lambda r: (r.question_id.values, r.problem_id.values, r.correct.values,
                                                   r.elapsed_time.values, r.skill.values))

    print(group)

    print("splitting")
    train, test = train_test_split(group, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print("train size: ", train.shape, "validation size: ", val.shape)

    train_dataset = DKTDataset(train.values, max_seq=max_questions)
    val_dataset = DKTDataset(val.values, max_seq=max_questions)
    test_dataset = DKTDataset(test.values, max_seq=max_questions)
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
