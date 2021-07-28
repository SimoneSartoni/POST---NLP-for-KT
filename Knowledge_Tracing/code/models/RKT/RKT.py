import os
from abc import abstractmethod, ABC
import copy
import numpy as np
import psutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from Knowledge_Tracing.code.evaluation.metrics_RKT import metrics_RKT

from Knowledge_Tracing.code.models.RKT.multi_head_attention import MultiHeadedAttention


def future_mask(seq_length):
    mask = np.triu(np.ones((1, seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(mask)


def clone(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])


class RKT(nn.Module):
    def __init__(self, num_items,  embed_size, num_attn_layers, num_heads,
                 encode_pos, max_pos, drop_prob, l1, l2):
        """Self-attentive knowledge tracing.
        Arguments:
            num_items (int): number of items
            num_skills (int): number of skills
            embed_size (int): input embedding and attention dot-product dimension
            num_attn_layers (int): number of attention layers
            num_heads (int): number of parallel attention heads
            encode_pos (bool): if True, use relative position embeddings
            max_pos (int): number of position embeddings to use
            drop_prob (float): dropout probability
        """
        super(RKT, self).__init__()
        self.embed_size = embed_size
        self.encode_pos = encode_pos

        self.item_embeds = nn.Embedding(num_items + 1, embed_size , padding_idx=0)
        # self.skill_embeds = nn.Embedding(num_skills + 1, embed_size // 2, padding_idx=0)

        self.pos_key_embeds = nn.Embedding(max_pos, embed_size // num_heads)
        self.pos_value_embeds = nn.Embedding(max_pos, embed_size // num_heads)

        self.lin_in = nn.Linear(2*embed_size, embed_size)
        self.attn_layers = clone(MultiHeadedAttention(embed_size, num_heads, drop_prob), num_attn_layers)
        self.dropout = nn.Dropout(p=drop_prob)
        self.lin_out = nn.Linear(embed_size, 1)
        self.l1 = nn.Parameter(torch.tensor(l1))
        self.l2 = nn.Parameter(torch.tensor(l2))
        self.pro_pro_sparse = None

    def set_pro_pro_sparse(self, pro_pro_sparse):
        self.pro_pro_sparse = pro_pro_sparse

    def get_inputs(self, item_inputs, label_inputs):
        item_inputs = self.item_embeds(item_inputs)
        # skill_inputs = self.skill_embeds(skill_inputs)
        label_inputs = label_inputs.unsqueeze(-1).float()

        inputs = torch.cat([item_inputs, item_inputs], dim=-1)
        inputs[..., :self.embed_size] *= label_inputs
        inputs[..., self.embed_size:] *= 1 - label_inputs
        return inputs

    def get_query(self, item_ids):
        item_ids = self.item_embeds(item_ids)
        # skill_ids = self.skill_embeds(skill_ids)
        query = torch.cat([item_ids], dim=-1)
        return query

    def forward(self, item_inputs, label_inputs, item_ids, rel, timestamp):
        inputs = self.get_inputs(item_inputs, label_inputs)

        inputs = F.relu(self.lin_in(inputs))

        query = self.get_query(item_ids)

        mask = future_mask(inputs.size(-2))
        if inputs.is_cuda:
            mask = mask.cuda()
        outputs, attn = self.attn_layers[0](query, inputs, inputs, rel, self.l1, self.l2, timestamp, self.encode_pos,
                                                   self.pos_key_embeds, self.pos_value_embeds, mask)
        outputs = self.dropout(outputs)
        for l in self.attn_layers[1:]:
            residual, attn = l(query, outputs, outputs, rel, self.l1, self.l2, self.encode_pos, timestamp, self.pos_key_embeds,
                         self.pos_value_embeds, mask)
            outputs = self.dropout(outputs + F.relu(residual))

        return self.lin_out(outputs), attn

    def train(self, train_data, val_data, pro_num, corr_data, timestamp, timespan, model, optimizer, logger, saver,
              num_epochs, batch_size, grad_clip):
        """Train SAKT model.
        Arguments:
            train_data (list of tuples of torch Tensor)
            val_data (list of tuples of torch Tensor)
            model (torch Module)
            optimizer (torch optimizer)
            logger: wrapper for TensorboardX logger
            saver: wrapper for torch saving
            num_epochs (int): number of epochs to train for
            batch_size (int)
            grad_clip (float): max norm of the gradients
        """

        params = {'batch_size': batch_size,
                  'shuffle': True}
        process = psutil.Process(os.getpid())
        print('entered train', process.memory_info().rss)
        criterion = nn.BCEWithLogitsLoss()
        step = 0
        metrics = metrics_RKT()
        test_generator = torch.utils.data.DataLoader(val_data, **params)
        print('PB memory used: ', process.memory_info().rss)
        for epoch in range(num_epochs):
            training_generator = torch.utils.data.DataLoader(train_data, **params)
            print("in epoch" + str(epoch))
            print("Prepare batches train")
            # train_batches = prepare_batches(train_data, batch_size)
            print("Prepare batches val")
            # val_batches = prepare_batches(val_data, batch_size)
            i = 0
            # Training
            for data, labels in training_generator:
                item_inputs, label_inputs, item_ids, timestamp = data

                # rel = compute_corr(item_inputs, item_ids, corr_data)
                """rel = torch.Tensor(self.pro_pro_sparse[(item_ids - 1).cpu().unsqueeze(1).repeat(1, item_ids.shape[-1], 1), (
                            item_inputs - 1).cpu().unsqueeze(-1).repeat(1, 1, item_inputs.shape[-1])]).cuda()"""
                rel = self.pro_pro_sparse.tocsr()[item_ids, :].dot(
                    self.pro_pro_sparse.tocsr().getrow(target_problem).transpose())
                item_scores = item_scores.transpose().todense().dot(corrects)

                time = computeRePos(timestamp, timespan)
                # skill_inputs = skill_inputs.cuda()
                # skill_ids = skill_ids.cuda()
                # item_ids = item_ids.cuda()
                preds, weights = model(item_inputs, label_inputs, item_ids, rel, time)
                loss = compute_loss(preds, labels, criterion)
                preds = torch.sigmoid(preds).detach().cpu()
                train_auc, train_acc = compute_auc(preds, labels.cpu())
                model.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                step += 1
                metrics.store({'loss/train': loss.item()})
                metrics.store({'auc/train': train_auc})

                # print(step)
                if step % 1000 == 0:
                    print(metrics.average())
                    print(step)

                    # weights = {"weight/" + name: param for name, param in model.named_parameters()}
                    # grads = {"grad/" + name: param.grad
                    #         for name, param in model.named_parameters() if param.grad is not None}
                    # logger.log_histograms(weights, step)
                    # logger.log_histograms(grads, step)
            # Logging
            torch.save(weights, 'weight_tensor_rel')
            # Validation

            model.eval()
            for data, labels in test_generator:
                item_inputs, label_inputs, item_ids, timestamp = data
                # rel = compute_corr(item_inputs, item_ids, corr_data)
                rel = torch.Tensor(corr_data[(item_ids - 1).cpu().unsqueeze(1).repeat(1, item_ids.shape[-1], 1), (
                            item_inputs - 1).cpu().unsqueeze(-1).repeat(1, 1, item_inputs.shape[-1])]).cuda()
                time = computeRePos(timestamp, timespan)
                with torch.no_grad():
                    preds, weights = model(item_inputs, label_inputs, item_ids, rel, time)
                    preds = torch.sigmoid(preds).cpu()
                val_auc, val_acc = compute_auc(preds, labels.cpu())
                metrics.store({'auc/val': val_auc, 'acc/val': val_acc})
                gc.collect()
            model.train()

            # Save model

            average_metrics = metrics.average()
            logger.log_scalars(average_metrics, step)
            print(average_metrics)

            stop = saver.save(average_metrics['auc/val'], model)
            if stop:
                break


