import scipy.io.wavfile as wavfile
import numpy as np
import os
from collections import deque
import argparse

parser = argparse.ArgumentParser(description = "Make a numpy data from binary file")
parser.add_argument("-f", "--file", metavar="WAV", help = "Path to your corresponding wav file")
args = vars(parser.parse_args())

SR = int(os.environ.get('SR', '16000'))
threshold = int(os.environ.get('THRESHOLD', '1000'))
minus = os.environ.get('MINUS', '0').lower() in ('true', '1', 't')
plus = os.environ.get('PLUS', '1').lower() in ('true', '1', 't')
L = int(os.environ.get('L', '1000'))


if not os.path.exists("data/npy"):
    os.mkdir("data/npy")

wf = args["file"]
print("file: {}".format(wf))
_, data = wavfile.read("data/wav/{}".format(wf))
bn, _ = os.path.splitext(wf)

bit = []
with open("data/binary/{}_SR{}_threshold{}".format(bn, SR, threshold),"rb") as f:
    for b in f.read():
        bit.append(b)

y = deque([0.0]*L)
S = 0.0
numpydata = []
for cur in range(len(bit)):
    popped = y.popleft()
    if cur-L >= 0:
        if bit[cur-L]:
            S-=popped
        else:
            S+=popped
    if bit[cur]:
        y.append(data[cur+1]/threshold-S)
        S+=y[-1]
    else:
        y.append(S-data[cur+1]/threshold)
        S-=y[-1]
    numpydata.append(list(y))

np.savez_compressed("data/npy/{}_SR{}_threshold{}_L{}".format(bn, SR, threshold, L), np.array(numpydata))
