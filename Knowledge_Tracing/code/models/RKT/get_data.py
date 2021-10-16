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

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

print(torch.cuda.is_available())
dt = datetime.utcnow()


def get_corr_data_assistments(pro_num):
    pro_pro_sparse = sparse.load_npz('../input/assesments-12-13-precessed-data/pro_pro_existing_words_only.npz')
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


def get_data_assistments(batch_size=64):
    """Extract sequences from dataframe.
    Arguments:
        batch_size (int): batch_size
    """

    params = {'batch_size': batch_size,
              'shuffle': True}
    process = psutil.Process(os.getpid())
    gc.enable()
    data = np.load('../input/assesments-12-13-precessed-data/2012-2013-data-with-predictions-4-final.csv.npz')
    y, skill, problem, timestamps, real_len = data['y'], data['skill'], data['problem'], data['time'], data['real_len']
    skill_num, pro_num = data['skill_num'], data['problem_num']
    print(pro_num)
    item_ids = [torch.tensor(i).type(torch.cuda.LongTensor) for i in problem]
    timestamp = [
        torch.tensor([(t - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's') for t in timestamp]).type(
            torch.cuda.LongTensor) for timestamp in timestamps]
    labels = [torch.tensor(i).type(torch.cuda.LongTensor) for i in y]
    item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long).cuda(), i))[:-1] for i in item_ids]
    # skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s))[:-1] for s in skill_ids]
    label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long).cuda(), l))[:-1] for l in labels]

    batches = list(zip(item_inputs, label_inputs, item_ids, timestamp, labels))
    seq_lists = list(zip(*batches))
    inputs_and_ids = [pad_sequence(seqs, batch_first=True, padding_value=0)
                      for seqs in seq_lists[0:4]]
    labels = pad_sequence(seq_lists[-1], batch_first=True, padding_value=-1)  # Pad labels with -1
    train_data, test_data, training_labels, test_labels = train_test_split(data=list(zip(*inputs_and_ids)),
                                                                           labels=labels, split=0.8)
    training_set = Dataset(train_data, training_labels)
    # training_generator = torch.utils.data.DataLoader(training_set, **params)
    test_set = Dataset(test_data, test_labels)
    # test_generator = torch.utils.data.DataLoader(test_set, **params)
    # validation_set = Dataset(val_data, val_labels)
    # validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    return training_set, test_set, pro_num, timestamps


def train_test_split(data, labels, split=0.8):
    n_samples = len(data)
    # x is your dataset
    training_data, test_data = data[:int(n_samples * split)], data[int(n_samples * split):]
    training_labels, test_labels = labels[:int(n_samples * split)], labels[int(n_samples * split):]
    return training_data, test_data, training_labels, test_labels


def compute_auc(preds, labels):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    if len(torch.unique(labels)) == 1:  # Only one class
        auc = accuracy_score(labels, preds.round())
        acc = auc
    else:
        preds2 = np.array(preds, dtype=float)
        preds2[preds2 >= 0.5] = 1.0
        preds2[preds2 < 0.5] = 0.0
        auc = roc_auc_score(y_true=labels, y_score=preds2)
        acc = accuracy_score(labels, preds.round())
        # print(str(i)+str(acc))
    return auc, acc


def compute_loss(predictions, labels, criterion):
    predictions = predictions[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    return criterion(predictions, labels)


def computeRePos(time_seq, time_span):
    batch_size = time_seq.shape[0]
    size = time_seq.shape[1]

    time_matrix = (
        torch.abs(torch.unsqueeze(time_seq, axis=1).repeat(1, size, 1).reshape((batch_size, size * size, 1)) - \
                  torch.unsqueeze(time_seq, axis=-1).repeat(1, 1, size, ).reshape((batch_size, size * size, 1))))

    # time_matrix[time_matrix>time_span] = time_span
    time_matrix = time_matrix.reshape((batch_size, size, size))

    return time_matrix
