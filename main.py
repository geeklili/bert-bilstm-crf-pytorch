# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import argparse
from torch.utils import data
from crf import Bert_BiLSTM_CRF
from utils import NerDataset, pad, VOCAB, tokenizer, tag2idx, idx2tag


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def train(model, iterator, optimizer, criterion, device):
    model.train()
    loss_pre = 1e10
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch
        x = x.to(device)
        y = y.to(device)
        _y = y  # for monitoring
        optimizer.zero_grad()
        loss = model.neg_log_likelihood(x, y)  # logits: (N, T, VOCAB), y: (N, T)

        # logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        # y = y.view(-1)  # (N*T,)
        # writer.add_scalar('data/loss', loss.item(), )
        # loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        if loss.item() < loss_pre:
            torch.save(model.state_dict(), './model/saved/bbc_model.pt')
            loss_pre = loss.item()

        if i == 0:
            print("=====sanity check======")
            # print("words:", words[0])
            print("x:", x.cpu().numpy()[0][:seqlens[0]])
            # print("tokens:", tokenizer.convert_ids_to_tokens(x.cpu().numpy()[0])[:seqlens[0]])
            print("is_heads:", is_heads[0])
            print("y:", _y.cpu().numpy()[0][:seqlens[0]])
            print("tags:", tags[0])
            print("seqlen:", seqlens[0])
            print("=======================")

        # monitoring
        if i % 10 == 0:
            print(f"step: {i}, loss: {loss.item()}")


def eval(model, iterator, device):
    model.eval()
    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch
            x = x.to(device)
            # y = y.to(device)

            _, y_hat = model(x)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    ## gets results and save
    with open("temp", 'w', encoding='utf-8') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds) == len(words.split()) == len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write(f"{w} {t} {p}\n")
            fout.write("\n")

    ## calc metric
    y_true = np.array(
        [tag2idx[line.split()[1]] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])
    y_pred = np.array(
        [tag2idx[line.split()[2]] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])

    num_proposed = len(y_pred[y_pred > 1])
    num_correct = (np.logical_and(y_true == y_pred, y_true > 1)).astype(np.int).sum()
    num_gold = len(y_true[y_true > 1])

    print(f"num_proposed:{num_proposed}")
    print(f"num_correct:{num_correct}")
    print(f"num_gold:{num_gold}")
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        if precision * recall == 0:
            f1 = 1.0
        else:
            f1 = 0

    # final = f + ".P%.2f_R%.2f_F%.2f" %(precision, recall, f1)
    # with open(final, 'w', encoding='utf-8') as fout:
    #     result = open("temp", "r", encoding='utf-8').read()
    #     fout.write(f"{result}\n")
    #
    #     fout.write(f"precision={precision}\n")
    #     fout.write(f"recall={recall}\n")
    #     fout.write(f"f1={f1}\n")
    #
    # os.remove("temp")

    print("precision=%.2f" % precision)
    print("recall=%.2f" % recall)
    print("f1=%.2f" % f1)
    return precision, recall, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true", default=True)
    parser.add_argument("--top_rnns", dest="top_rnns", action="store_true", default=True)
    parser.add_argument("--model_dir", type=str, default="./model/saved/")
    # parser.add_argument("--trainset", type=str, default="./processed/processed_training_bio.txt")
    # parser.add_argument("--validset", type=str, default="./processed/processed_dev_bio.txt")
    parser.add_argument("--trainset", type=str, default="./data/example.train")
    parser.add_argument("--validset", type=str, default="./data/example.dev")
    # parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--device", default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    pars = parser.parse_args()

    if not os.path.exists(pars.model_dir):
        os.makedirs(pars.model_dir)

    model = Bert_BiLSTM_CRF(tag2idx).to(pars.device)
    print('Initial model Done')
    # model = nn.DataParallel(model)

    train_dataset = NerDataset(pars.trainset)
    eval_dataset = NerDataset(pars.validset)
    print('Load Data Done')

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=pars.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=pars.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr=pars.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print('Start Train...,')
    f1_max = 0
    train_or_not = 0
    for epoch in range(1, pars.n_epochs + 1):
        print(f"=========train at epoch={epoch}=========")  # 每个epoch对dev集进行测试
        train(model, train_iter, optimizer, criterion, pars.device)

        print(f"=========eval at epoch={epoch}=========")
        model.load_state_dict(torch.load(pars.model_dir + f"bbc_model.pt", map_location=pars.device))
        precision, recall, f1 = eval(model, eval_iter, pars.device)
        if f1 >= f1_max and train_or_not:
            torch.save(model.state_dict(), pars.model_dir + f"bbc_model.pt")
            f1_max = f1
            print(f"weights were saved to bbc_model.pt")
