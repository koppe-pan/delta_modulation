from rnn_model import fittingRNN
import torch
import torch.nn as nn
from torch import cuda
from torch import optim
import os
import numpy as np
import argparse
gpu_id = None

SR = int(os.environ.get('SR', '16000'))
threshold = int(os.environ.get('THRESHOLD', '1000'))
L = int(os.environ.get('L', '3'))

parser = argparse.ArgumentParser(description = "Make a wav data from numpy file")
parser.add_argument("-f", "--file", metavar="WAV", help = "Path to your corresponding original wav file")
args = vars(parser.parse_args())
wf = args["file"]
print("file: {}".format(wf))
bn, _ = os.path.splitext(wf)

# test
model = fittingRNN()
model.load_state_dict(torch.load("trained_model/fitting_model_rnn.model"))
model.eval()
bits = []
with open("data/binary/{}_SR{}_threshold{}".format(bn, SR, threshold),"rb") as f:
    for b in f.read():
        bits.append(b)


numpydata = []
x = torch.tensor([0.0]*(L*2))
for i in range(1,L):
    y = model(x)
    cur = list(map(lambda l: l.item(), y[0][L:L*2]))
    numpydata.append(cur)
    x = torch.tensor([0.0]*(L-i) + bits[0:i] + cur)

for i in range(len(bits)-L+1):
    if len(x)%L != 0:
        numpydata.append([0.0]*L)
        break
    y = model(x)
    cur = list(map(lambda l: l.item(), y[0][L:L*2]))
    numpydata.append(cur)
    x = torch.tensor(bits[i:L+i] + cur)

np.savez_compressed("data/npy/{}_SR{}_threshold{}_L{}_transformed".format(bn, SR, threshold, L), np.array(numpydata))
