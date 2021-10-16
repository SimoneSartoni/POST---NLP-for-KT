import argparse
import gc
import os

import psutil
from torch.optim import Adam

from Knowledge_Tracing.code.models.RKT.get_data import get_corr_data_assistments, get_data_assistments
from Knowledge_Tracing.code.models.RKT.utils import Logger, Saver
from code.models.RKT.RKT import RKT
from code.models.RKT.RKT_train import train


def RKT_run():
    parser = argparse.ArgumentParser(description='Train RKT.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--logdir', type=str, default='runs/rkt')
    parser.add_argument('--savedir', type=str, default='./')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=50)
    parser.add_argument('--embed_size', type=int, default=64)
    parser.add_argument('--num_attn_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--encode_pos', action='store_true')
    parser.add_argument('--max_pos', type=int, default=10)
    parser.add_argument('--drop_prob', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--l1', type=float, default=0.5)
    parser.add_argument('--l2', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=10)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--timespan', default=100000, type=int)

    args = parser.parse_args(args=[])

    # full_df = pd.read_csv('./', sep=",")
    # train_df = pd.read_csv('../../KT-GAT/data/ed_net2_train.csv', sep=",")
    # test_df = pd.read_csv('../../KT-GAT/data/ed_net2_test.csv', sep=",")
    # # train_data_file = '../KT-GAT/data/ed_net.csv'
    # print(len(train_data))
    process = psutil.Process(os.getpid())
    gc.enable()
    memory_0 = process.memory_info().rss
    train_data, test_data, pro_num, timestamp = get_data_assistments(batch_size=args.batch_size)
    print("Corr_data computation")
    pro_num = 179950
    corr_data = get_corr_data_assistments(pro_num)
    print("Corr_data computed")
    print(process.memory_info().rss - memory_0)
    print('Memory for leading train_data and corr_data: ', memory_0)
    # num_items = int(full_df["item_id"].max() + 1)
    # num_skills = int(full_df["skill_id"].max() + 1)
    num_items = pro_num
    models = []
    optimizers = []
    l1s = [0.0, 0.6, 0.8, 1.0]
    l2s = [0.0, 2.77, 6.0, 19.0]
    l3s = [0.0]
    for i in l1s:
        for j in l2s:
            for k in l3s:
                l1 = i / 1.0
                l2 = j / 1.0
                l3 = k / 1.0
                model = RKT(num_items, args.embed_size, args.num_attn_layers, args.num_heads,
                            args.encode_pos, args.max_pos, args.drop_prob, l1, l2, l3).cuda()
                models.append(model)
                optimizer = Adam(model.parameters(), lr=args.lr)
                optimizers.append(optimizer)
    memory_1 = process.memory_info().rss
    print('Memory for model definition: ', memory_1 - memory_0)

    # Reduce batch size until it fits on GPU
    while True:
        # try:
        # Train
        param_str = (f'{args.dataset},'
                     f'batch_size={args.batch_size},'
                     f'max_length={args.max_length},'
                     f'encode_pos={args.encode_pos},'
                     f'max_pos={args.max_pos}')
        logger = Logger(os.path.join(args.logdir, param_str))
        saver = Saver(args.savedir, param_str, patience=100)
        print('before train', process.memory_info().rss)
        train(train_data, test_data, pro_num, corr_data, timestamp, args.timespan, models, optimizers, logger, saver,
              args.num_epochs,
              args.batch_size, args.grad_clip)
        break
        # except RuntimeError:
        #   args.batch_size = args.batch_size // 2
        #  print(RuntimeError)
        # print(f'Batch does not fit on gpu, reducing size to {args.batch_size}')

    logger.close()

    param_str = (f'{args.dataset},'
                 f'batch_size={args.batch_size},'
                 f'max_length={args.max_length},'
                 f'encode_pos={args.encode_pos},'
                 f'max_pos={args.max_pos}')
    saver = Saver(args.savedir, param_str)
    model = saver.load()

    # Predict on test set
    print("pre eval")
    model.eval()
    print("post eval")
    correct = np.empty(0)
    i = 0
    test_preds = np.empty(0)
    for data, labels in test_data:
        item_inputs, label_inputs, item_ids, timestamp = data
        rel = torch.Tensor(corr_data[(item_ids - 1).cpu().unsqueeze(1).repeat(1, item_ids.shape[-1], 1), (
                    item_inputs - 1).cpu().unsqueeze(-1).repeat(1, 1, item_inputs.shape[-1])]).cuda()
        # skill_inputs = skill_inputs.cuda()
        time = computeRePos(timestamp, args.timespan)
        # skill_ids = skill_ids.cuda()
        with torch.no_grad():
            preds, weights = model(item_inputs, label_inputs, item_ids, rel, time)
            preds = torch.sigmoid(preds[labels >= 0]).flatten().cpu().numpy()
            test_preds = np.concatenate([test_preds, preds])
            if (i % 100):
                print(test_preds.shape)
        labels = labels[labels >= 0].float()
        correct = np.concatenate([correct, labels.cpu()])
        if (i % 100):
            print(correct.shape)
        i += 1

    print(correct.shape)
    print(test_preds.shape)
    test_preds_inverted = []
    correct_inverted = []
    for answer in test_preds:
        test_preds_inverted.append(1 - answer)
    for answer in correct:
        correct_inverted.append(1 - answer)
    print("auc_test = ", roc_auc_score(correct, test_preds))
    # print("acc_test = ", accuracy_score(correct, test_preds))