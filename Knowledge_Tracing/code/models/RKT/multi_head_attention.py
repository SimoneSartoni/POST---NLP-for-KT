import copy

import torch.nn as nn

from Knowledge_Tracing.code.models.RKT.attention import attention
from Knowledge_Tracing.code.models.RKT.relative_attention import relative_attention


def clone(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, total_size, num_heads, drop_prob):
        super(MultiHeadedAttention, self).__init__()
        assert total_size % num_heads == 0
        self.total_size = total_size
        self.head_size = total_size // num_heads
        self.num_heads = num_heads
        self.linear_layers = clone(nn.Linear(total_size, total_size), 3)
        self.dropout = nn.Dropout(p=drop_prob)
        self.prob_attn = None

    def forward(self, query, key, value, rel, l1, l2, timestamp, encode_pos, pos_key_embeds, pos_value_embeds,
                mask=None):
        batch_size, seq_length = query.shape[:2]

        # Apply mask to all heads
        if mask is not None:
            mask = mask.unsqueeze(1)

        # Project inputs
        rel = rel.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        timestamp = timestamp.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        query, key, value = [l(x).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # Apply attention
        if encode_pos:
            out, self.prob_attn = relative_attention(query, key, value, rel, l1, l2, timestamp, pos_key_embeds,
                                                     pos_value_embeds, mask, self.dropout)
        else:
            out, self.prob_attn = attention(query, key, value, rel, l1, l2, timestamp, mask, self.dropout)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.total_size)
        return out, self.prob_attn
