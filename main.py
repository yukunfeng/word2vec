# coding: utf-8
import argparse
import time
import datetime
import math
import os
import torch
import torch.nn as nn

import data
from model import CBOW

parser = argparse.ArgumentParser(description='CBOW')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--lr', type=float, default=0.1,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=200, metavar='N',
                    help='batch size')
parser.add_argument('--windows', type=str, default="5 5",
                    help='left and right window size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='cuda')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--cluster_path', type=str, default=None,
                    help='path of word cluster')
parser.add_argument('--freq_thre', type=int, default=-1,
                    help='freq threshould')
parser.add_argument('--max_gram_n', type=int, default=3,
                    help='ngram n')
parser.add_argument('--input_freq', type=int, default=None,
                    help='freq threshould for input word')
parser.add_argument('--note', type=str, default="",
                    help='extra note in final one-line result output')
parser.add_argument('--use_ngram', action='store_true',
                    help='use ngram')

args = parser.parse_args()
print(args)

args.windows = args.windows.split()
args.left_win = int(args.windows[0])
args.right_win = int(args.windows[1])

torch.manual_seed(args.seed)
device = torch.device(args.device)

###############################################################################
# Load data
###############################################################################

input_extra_unk = "<input_extra_unk>"
corpus = data.Corpus(
    args.data,
    cluster_path=args.cluster_path,
    freq_thre=args.freq_thre,
    use_ngram=args.use_ngram,
    max_gram_n=args.max_gram_n,
    input_freq=args.input_freq,
    input_extra_unk=input_extra_unk
)


def batchify(data):
    data_size = data.size(0) - args.left_win - args.right_win
    context_size = args.left_win + 1 + args.right_win
    cbow_data = torch.zeros((data_size, context_size), dtype=data.dtype)
    for i in range(0, data.size(0) - context_size + 1):
        cbow_data[i] = data[i: i + context_size]
    target_data = cbow_data[:, args.left_win]
    if args.right_win != 0:
        cbow_data = torch.cat([cbow_data[:, 0:args.left_win], cbow_data[:, args.left_win+1:]], 1)
    else:
        cbow_data = cbow_data[:, 0:args.left_win]
    return cbow_data.to(device), target_data.to(device)


def get_batch(cbow_data, target, i):
    cbow_data_batch = cbow_data[i:i+args.batch_size, :]
    target_batch = target[i:i+args.batch_size]
    return cbow_data_batch, target_batch

full_cbow, full_target = batchify(corpus.train)
# get fixed input data
fixed_cbow, fixed_target = batchify(corpus.train_fixed)
full_vocab_size = len(corpus.dictionary)

cluster_train = None
if args.cluster_path is not None:
    cluster_cbow, cluster_target = batchify(corpus.cluster_train)

# Define model
model = CBOW(
    full_vocab_size,
    args.emsize,
    args.left_win + args.right_win
).to(device)


# Training
batch_number = full_cbow.size(0) // args.batch_size
end_position = batch_number * args.batch_size - args.batch_size + 1

criterion = nn.CrossEntropyLoss()

print(f"start training")
for epoch in range(1, args.epochs+1):
    model.train()
    total_loss = 0.
    start_time = datetime.datetime.now()
    for i in range(0, end_position, args.batch_size):
        full_cbow_batch, full_target_batch = get_batch(full_cbow, full_target, i)

        model.zero_grad()
        output = model(full_cbow_batch)
        loss = criterion(output.view(-1, full_vocab_size), full_target_batch)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #  torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            if p.grad is not None:
                p.data.add_(-args.lr, p.grad.data)

        total_loss += loss.item()
    end_time = datetime.datetime.now()
    time_elapsed = f"{end_time - start_time}"
    print(f"epoch {epoch} loss {total_loss:5.2f} time_elapsed: {time_elapsed}")
