import Knowledge_Tracing.code.models.complex_models.config as config
from Knowledge_Tracing.code.utils.utils import try_parsing_date
from datetime import datetime

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gc
from sklearn.model_selection import train_test_split


class DKTDataset(Dataset):
    def __init__(self, group, max_seq=100):
        self.samples = group
        self.max_seq = max_seq
        self.data = []

        for que, ans, res_time, exe_cat in self.samples:
            if len(que) >= self.max_seq:
                self.data.extend([(que[l:l + self.max_seq], ans[l:l + self.max_seq], res_time[l:l + self.max_seq],
                                   exe_cat[l:l + self.max_seq]) for
                                  l in range(len(que)) if l % self.max_seq == 0])
            elif self.max_seq > len(que) > 1:
                self.data.append((que, ans, res_time, exe_cat))
            else:
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        content_ids, answered_correctly, response_time, exe_category = self.data[idx]
        seq_len = len(content_ids)

        q_ids = np.zeros(self.max_seq, dtype=int)
        ans = np.zeros(self.max_seq, dtype=int)
        r_time = np.zeros(self.max_seq, dtype=int)
        exe_cat = np.zeros(self.max_seq, dtype=int)

        if seq_len >= self.max_seq:
            q_ids[:] = content_ids[-self.max_seq:]
            ans[:] = answered_correctly[-self.max_seq:]
            r_time[:] = response_time[-self.max_seq:]
            exe_cat[:] = exe_category[-self.max_seq:]
        else:
            q_ids[-seq_len:] = content_ids
            ans[-seq_len:] = answered_correctly
            r_time[-seq_len:] = response_time
            exe_cat[-seq_len:] = exe_category

        target_qids = q_ids[1:]
        label = ans[1:]

        input_ids = q_ids[:-1].copy()

        input_rtime = r_time[:-1].copy()

        input_cat = exe_cat[:-1].copy()

        input = {"input_ids": input_ids, "input_rtime": input_rtime.astype(np.int), "input_skills": input_cat}

        return input, target_qids, label


def get_dataloaders(nrows=10000):
    dtypes = {'user_id': 'int32', 'problem_id': 'int64',
              'correct': 'float64', 'skill': "string",
              'start_time': "string", 'end_time': "string"}

    print("loading csv.....")
    train_df = pd.read_csv(config.TRAIN_FILE, dtype=dtypes, nrows=nrows)
    print("shape of dataframe :", train_df.shape)

    train_df.fillna("no_skill", inplace=True)
    print("shape after drop no skill:", train_df.shape)

    # Step 1.2 - Remove users with a single answer
    train_df = train_df.groupby('user_id').filter(lambda q: len(q) > 1).copy()
    print("shape after at least 2 interactions:", train_df.shape)

    # Step 2 - Enumerate skill id
    train_df['skill'], _ = pd.factorize(train_df['skill'], sort=True)
    print("shape after factorize:", train_df.shape)

    train_df['start_time'] = [try_parsing_date(x) for x in train_df['start_time']]
    train_df['end_time'] = [try_parsing_date(x) for x in train_df['end_time']]

    train_df["prior_question_elapsed_time"] = [datetime.strptime(end, '%Y-%m-%d %H:%M:%S').timestamp() -
                                               datetime.strptime(start, '%Y-%m-%d %H:%M:%S').timestamp()
                                               for start, end in
                                               list(zip(train_df['start_time'], train_df['end_time']))]

    train_df["prior_question_elapsed_time"].fillna(300, inplace=True)
    train_df["prior_question_elapsed_time"].clip(lower=0, upper=300, inplace=True)
    train_df["prior_question_elapsed_time"] = train_df["prior_question_elapsed_time"].astype(np.int)

    train_df["timestamp"] = [datetime.strptime(start, '%Y-%m-%d %H:%M:%S').timestamp()
                             for start in train_df['start_time']]
    train_df = train_df.sort_values(["timestamp"], ascending=True).reset_index(drop=True)
    ids = train_df['problem_id'].unique()
    n_ids = len(ids)
    n_skills = len(train_df['skill'].unique()) + 100
    print("no. of problems :", n_ids)
    print("no. of skills: ", n_skills)
    print("shape after exclusion:", train_df.shape)

    train_df['content_id'], _ = pd.factorize(train_df['problem_id'], sort=True)

    # grouping based on user_id to get the data supply
    print("Grouping users...")
    group = train_df[
        ["user_id", "content_id", "correct", "prior_question_elapsed_time", "skill"]] \
        .groupby("user_id") \
        .apply(lambda r: (r.content_id.values, r.correct.values, r.prior_question_elapsed_time.values, r.skill.values))

    print(group)

    print("splitting")
    train, test = train_test_split(group, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print("train size: ", train.shape, "validation size: ", val.shape)
    print()
    train_dataset = DKTDataset(train.values, max_seq=config.MAX_SEQ)
    val_dataset = DKTDataset(val.values, max_seq=config.MAX_SEQ)
    test_dataset = DKTDataset(test.values, max_seq=config.MAX_SEQ)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              num_workers=2,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            num_workers=2,
                            shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=config.BATCH_SIZE,
                             num_workers=2,
                             shuffle=False)
    del train_dataset, val_dataset, test_dataset
    gc.collect()
    return train_loader, val_loader, test_loader
