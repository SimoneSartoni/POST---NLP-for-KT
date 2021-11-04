import os

import numpy as np
import psutil
import gc
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy import sparse
import torch
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime

from Knowledge_Tracing.code.models.RKT.dataset import Dataset

dt = datetime.utcnow()


def get_corr_data_assistments(filepath='../input/assesments-12-13-precessed-data/pro_pro_existing_words_only.npz'):
    pro_pro_sparse = sparse.load_npz(filepath)
    pro_pro_coo = pro_pro_sparse.tocoo()
    # print(pro_skill_csr)
    pro_pro_dense = pro_pro_coo.toarray()
    return pro_pro_dense


def get_corr_data(pro_num):
    pro_pro_dense = np.zeros((pro_num, pro_num))
    pro_pro_ = open('../input/ednet-dataset/ednet_corr.csv')
    for i in pro_pro_:
        j = i.strip().split(',')
        pro_pro_dense[int(j[0])][int(j[1])] += int(float(j[2]))
    return pro_pro_dense


def get_data_assistments(batch_size=64, use_skills=True,
                         filepath="../input/assesments-12-13-precessed-data/2012-2013-data-with-predictions-4-final.csv.npz"):
    """Extract sequences from dataframe.
    Arguments:
        batch_size (int): batch_size
    """

    params = {'batch_size': batch_size,
              'shuffle': True}
    process = psutil.Process(os.getpid())
    gc.enable()
    data = np.load(filepath)
    y, skill, problem, timestamps, real_len = data['y'], data['skill'], data['problem'], data['time'], data['real_len']
    skill_num, pro_num = data['skill_num'], data['problem_num']
    print(pro_num)
    item_ids = [torch.tensor(i).type(torch.cuda.LongTensor) for i in problem]
    skill_ids = [torch.tensor(i).type(torch.cuda.LongTensor) for i in skill]
    timestamp = [
        torch.tensor([(t - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's') for t in timestamp]).type(
            torch.cuda.LongTensor) for timestamp in timestamps]
    labels = [torch.tensor(i).type(torch.cuda.LongTensor) for i in y]
    item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long).cuda(), i))[:-1] for i in item_ids]
    if use_skills:
        skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long).cuda(), s))[:-1] for s in skill_ids]
    label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long).cuda(), l))[:-1] for l in labels]
    if use_skills:
        batches = list(zip(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, timestamp, labels))
    else:
        batches = list(zip(item_inputs, label_inputs, item_ids, timestamp, labels))
    seq_lists = list(zip(*batches))
    inputs_and_ids = [pad_sequence(seqs, batch_first=True, padding_value=0)
                      for seqs in seq_lists[0:-1]]
    labels = pad_sequence(seq_lists[-1], batch_first=True, padding_value=-1)  # Pad labels with -1
    train_data, test_data, train_labels, test_labels = train_test_split(data=list(zip(*inputs_and_ids)),
                                                                        labels=labels, split=0.8)
    train_data, val_data, train_labels, val_labels = train_test_split(data=train_data,
                                                                      labels=train_labels, split=0.75)
    training_set = Dataset(train_data, train_labels)
    test_set = Dataset(test_data, test_labels)
    validation_set = Dataset(val_data, val_labels)

    if use_skills:
        return training_set, validation_set, test_set, pro_num, skill_num, timestamps
    return training_set, validation_set, test_set, pro_num, timestamps


def train_test_split(data, labels, split=0.8):
    n_samples = len(data)
    # x is your dataset
    training_data, test_data = data[:int(n_samples * split)], data[int(n_samples * split):]
    training_labels, test_labels = labels[:int(n_samples * split)], labels[int(n_samples * split):]
    return training_data, test_data, training_labels, test_labels


def compute_metrics(outputs, labels):
    outputs = outputs[labels >= 0].float()
    labels = labels[labels >= 0].float()
    if len(torch.unique(labels)) == 1:  # Only one class
        auc = accuracy_score(labels, outputs.round())
        acc = auc
        print("wrong, it entered here")
    else:
        predictions = [1.0 if output >= 0.5 else 0.0 for output in outputs]
        auc = roc_auc_score(y_true=labels, y_score=outputs)
        acc = accuracy_score(labels, predictions)
    return auc, acc


def compute_loss(predictions, labels, criterion):
    predictions = predictions[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    return criterion(predictions, labels)


def computeRePos(time_seq, time_span=0):
    batch_size = time_seq.shape[0]
    size = time_seq.shape[1]

    time_matrix = (
        torch.abs(torch.unsqueeze(time_seq, axis=1).repeat(1, size, 1).reshape((batch_size, size * size, 1)) - \
                  torch.unsqueeze(time_seq, axis=-1).repeat(1, 1, size, ).reshape((batch_size, size * size, 1))))

    if time_span > 0:
        time_matrix[time_matrix > time_span] = time_span
    time_matrix = time_matrix.reshape((batch_size, size, size))
    return time_matrix
