import numpy as np
import os
import scipy.io.wavfile as wavfile
import argparse

parser = argparse.ArgumentParser(description = "Make a wav data from numpy file")
parser.add_argument("-f", "--file", metavar="WAV", help = "Path to your corresponding original wav file")
args = vars(parser.parse_args())

wf = args["file"]
print("file: {}".format(wf))
bn, _ = os.path.splitext(wf)

SR = int(os.environ.get('SR', '16000'))
threshold = int(os.environ.get('THRESHOLD', '1000'))
minus = os.environ.get('MINUS', '0').lower() in ('true', '1', 't')
plus = os.environ.get('PLUS', '1').lower() in ('true', '1', 't')
L = int(os.environ.get('L', '1'))

bits = []
with open("data/binary/{}_SR{}_threshold{}".format(bn, SR, threshold),"rb") as f:
    for b in f.read():
        bits.append(b)

loaded = np.load("data/npy/{}_SR{}_threshold{}_L{}.npz".format(bn, SR, threshold, L))['arr_0']

'''
ex.) let `L` be `3`
         `loaded` be `[0,0,0.2], [0,0.2,-0.3], [0.2,-0.3,3.2], [-0.3,3.2,1.1], ..., [-1.1,2,1]]`
         `bits` be `[1,0,1,1,1,0,1, ..., 1,0]`
then
```
y[0] = 0
y[1] = <[0,0,0.2], [0,0,1]> = 0.2
y[2] = <[0,0.2,-0.3], [0,1,-1]> = 0.5
y[3] = <[0.2,-0.3,3.2], [1,-1,1]> = 3.7
y[4] = <[-0.3,3.2,1.1], [-1,1,1]> = 4.6
```
where <,> is the inner product.
Each `bit` represents `1` with `1` and `-1` with `0`.
'''
y = [0]
for bit in range(len(bits)):
    S = 0
    for i in range(L):
        cur = bit-L+i
        if cur < 0: continue
        loaded_cur = loaded[bit-1][i]
        if bits[cur]:
            S+=loaded_cur
        else:
            S-=loaded_cur
    y.append(S)
if not os.path.exists("data/result"):
    os.mkdir("data/result")
wavfile.write("data/result/{}_SR{}_threshold{}_L{}.wav".format(bn, SR, threshold, L), SR, np.array(y))
