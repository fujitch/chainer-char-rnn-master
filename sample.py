#%%
import time
import math
import sys
import argparse
import cPickle as pickle
import codecs
import MeCab
import random

import numpy as np
from chainer import cuda, Variable, FunctionSet
import chainer.functions as F
from CharRNN import CharRNN, make_initial_state

sys.stdout = codecs.getwriter('utf_8')(sys.stdout)

#%% arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model',      type=str,   default='cv/numo/latest.chainermodel')
parser.add_argument('--vocabulary', type=str,   default='data/numo/vocab.bin')

parser.add_argument('--seed',       type=int,   default=123)
parser.add_argument('--sample',     type=int,   default=1)
parser.add_argument('--primetext',  type=str,   default='最終処分')
parser.add_argument('--length',     type=int,   default=2000)
parser.add_argument('--gpu',        type=int,   default=-1)

args = parser.parse_args()
m_owakati = MeCab.Tagger(r'-Owakati -d C:\Users\hori\workspace\encoder-decoder-sentence-chainer-master\mecab-ipadic-neologd')

np.random.seed(args.seed)

# load vocabulary
vocab = pickle.load(open(args.vocabulary, 'rb'))
ivocab = {}
for c, i in vocab.items():
    ivocab[i] = c

# load model
model = pickle.load(open(args.model, 'rb'))
n_units = model.embed.W.data.shape[1]

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    
def random_index_choice(p_list):
    base = random.random()
    sorted_list = []
    for p in p_list:
        sorted_list.append(p)
    sorted_list.sort()
    sorted_list.reverse()
    for p in sorted_list:
        if p > base:
            return prob_list.index(p)
        else:
            base -= p
    return prob_list.index(p)

# initialize generator
state = make_initial_state(n_units, batchsize=1, train=False)
if args.gpu >= 0:
    for key, value in state.items():
        value.data = cuda.to_gpu(value.data)

prev_char = np.array([0], dtype=np.int32)
if args.gpu >= 0:
    prev_char = cuda.to_gpu(prev_char)
sum_string = ''
if len(args.primetext) > 0:
    prime_list = m_owakati.parse(args.primetext).split(" ")
    prime_list = prime_list[:len(prime_list)-1]
    for i in prime_list:
        i = i.decode('utf-8')
        sum_string = sum_string + i
        prev_char = np.ones((1,), dtype=np.int32) * vocab[i]
        if args.gpu >= 0:
            prev_char = cuda.to_gpu(prev_char)

        state, prob = model.forward_one_step(prev_char, prev_char, state, train=False)

sum_ind1 = []
sum_ind2 = ''
for i in xrange(args.length):
    state, prob = model.forward_one_step(prev_char, prev_char, state, train=False)

    if args.sample > 0:
        probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
        probability /= np.sum(probability)
        prob_list = []
        for a_p in probability:
            prob_list.append(a_p)
        index = random_index_choice(prob_list)
        # index = np.random.choice(len(probability), p=prob_list)
    else:
        index = np.argmax(cuda.to_cpu(prob.data))
    sum_string = sum_string + ivocab[index]

    prev_char = np.array([index], dtype=np.int32)
    if args.gpu >= 0:
        prev_char = cuda.to_gpu(prev_char)
