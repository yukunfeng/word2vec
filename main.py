# coding: utf-8
import argparse
import time
import datetime
import math
import os
import torch
import torch.nn as nn

import data
import model

parser = argparse.ArgumentParser(description='CBOW')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--windows', type=str, default="2 0",
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

def batchify(data, bsz):
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


def batchify_ngram(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, nbatch, -1).t().contiguous()
    return data.to(device)


eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

# get fixed input data
fixed_train_data = batchify(corpus.train_fixed, args.batch_size)
fixed_val_data = batchify(corpus.valid_fixed, eval_batch_size)
fixed_test_data = batchify(corpus.test_fixed, eval_batch_size)

cluster_train = None
cluster_val = None
cluster_test = None
if args.cluster_path is not None:
    cluster_train = batchify(corpus.cluster_train, args.batch_size)
    cluster_val = batchify(corpus.cluster_valid, eval_batch_size)
    cluster_test = batchify(corpus.cluster_test, eval_batch_size)

if args.use_ngram:
    ngram_train = batchify_ngram(corpus.ngram_train, args.batch_size)
    ngram_val = batchify_ngram(corpus.ngram_valid, eval_batch_size)
    ngram_test = batchify_ngram(corpus.ngram_test, eval_batch_size)

    ngram_train_len = None
    ngram_valid_len = None
    ngram_test_len = None

