# Code from RKT train with few changes for performance
import gc
import os

import numpy as np
import psutil
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from Knowledge_Tracing.code.models.RKT.get_data import computeRePos, compute_loss, compute_auc, get_corr_data_assistments
from Knowledge_Tracing.code.models.RKT.utils import Metrics


def train(train_data, val_data, pro_num, corr_data, timestamp, timespan, models, optimizers, logger, saver, num_epochs,
          batch_size, grad_clip):
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
    metrics_list = []
    for model in models:
        metrics_list.append(Metrics())
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
            rel = torch.Tensor(corr_data[(item_ids - 1).cpu().unsqueeze(1).repeat(1, item_ids.shape[-1], 1), (
                        item_inputs - 1).cpu().unsqueeze(-1).repeat(1, 1, item_inputs.shape[-1])]).cuda()
            time = computeRePos(timestamp, timespan)
            # skill_inputs = skill_inputs.cuda()
            # skill_ids = skill_ids.cuda()
            # item_ids = item_ids.cuda()
            for i in range(len(models)):
                model = models[i]
                metrics = metrics_list[i]
                optimizer = optimizers[i]
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

        # Logging
        torch.save(weights, 'weight_tensor_rel')
        # Validation
        for model in models:
            model.eval()
        for data, labels in test_generator:
            for i in range(len(models)):
                model = models[i]
                metrics = metrics_list[i]
                item_inputs, label_inputs, item_ids, timestamp = data
                rel = torch.Tensor(corr_data[(item_ids - 1).cpu().unsqueeze(1).repeat(1, item_ids.shape[-1], 1), (
                            item_inputs - 1).cpu().unsqueeze(-1).repeat(1, 1, item_inputs.shape[-1])]).cuda()
                time = computeRePos(timestamp, timespan)
                with torch.no_grad():
                    preds, weights = model(item_inputs, label_inputs, item_ids, rel, time)
                    preds = torch.sigmoid(preds).cpu()
                val_auc, val_acc = compute_auc(preds, labels.cpu())
                metrics.store({'auc/val': val_auc, 'acc/val': val_acc})
                gc.collect()
        for i in range(len(models)):
            models[i].train()
            # Save model
            metrics = metrics_list[i]
            average_metrics = metrics.average()
            stop = saver.save(average_metrics['auc/val'], model)
            logger.log_scalars(average_metrics, step)
            print(average_metrics)
            if stop:
                break


