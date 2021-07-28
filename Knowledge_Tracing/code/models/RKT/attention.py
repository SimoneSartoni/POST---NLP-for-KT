import math


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def attention(query, key, value, rel, l1, l2, timestamp, mask=None, dropout=None):
    """Compute scaled dot product attention.
    """
    print(rel)
    rel = rel * mask.to(torch.float)  # future masking of correlation matrix.
    print(rel)
    rel_attn = rel.masked_fill(rel == 0, -10000)
    print(rel)
    rel_attn = nn.Softmax(dim=-1)(rel_attn)
    print(rel)
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
        time_stamp = torch.exp(-torch.abs(timestamp.float()))
        #
        time_stamp = time_stamp.masked_fill(mask, -np.inf)
    print(scores)
    prob_attn = F.softmax(scores, dim=-1)
    time_attn = F.softmax(time_stamp,dim=-1)
    prob_attn = (1-l2)*prob_attn+l2*time_attn
    # prob_attn = F.softmax(prob_attn + rel_attn, dim=-1)

    prob_attn = (1-l1) * prob_attn + l1 * rel_attn
    if dropout is not None:
        prob_attn = dropout(prob_attn)
    return torch.matmul(prob_attn, value), prob_attn
