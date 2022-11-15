import torch
from torch import nn, optim
from torch.nn import functional as F

import numpy as np
import pysptk as sptk
from scipy.io import wavfile
import os
from pysptk.synthesis import MLSADF, Synthesizer

srcspk = os.environ.get('SRC', 'src')
tgtspk = os.environ.get('TRG', 'trg')

fs = 16000
fftlen = 512
alpha = 0.42
dim = 25

n_units = 128

# mgc
class VCDNN(nn.Module):
    def __init__(self, dim=25, n_units=256):
        super(VCDNN, self).__init__()
        self.fc = nn.ModuleList([
                       nn.Linear(dim, n_units),
                       nn.Linear(n_units, n_units),
                       nn.Linear(n_units, dim)
        ])

    def forward(self, x):
        h1 = F.relu(self.fc[0](x))
        h2 = F.relu(self.fc[1](h1))
        h3 = self.fc[2](h2)
        return h3

    def get_predata(self, x):
        _x = torch.from_numpy(x.astype(np.float32))
        return self.forward(_x).detach().numpy()
model = VCDNN(dim,n_units)
_ = model.load_state_dict(torch.load("model/vcmodel.model"))
datalist = []
with open("conf/eval.list", "r") as f:
    for line in f:
        line = line.rstrip()
        datalist.append(line)



for d in datalist:
    with open("data/mgc/{}/{}.mgc".format(srcspk,d),"rb") as f:
        outfile = "data/result/{}_diff.wav".format(d)
        dat = np.fromfile(f,dtype="<f8",sep="")
        src_mgc = dat.reshape(len(dat)//dim,dim)
        conv_mgc = model.get_predata(src_mgc)

    fs, data = wavfile.read("data/wav/{}.wav".format(d))  # 入力音声そのものをもってくる
    data = data.astype(float)

    diff_mgc = conv_mgc - src_mgc  # 差分のフィルタを用意する

    # 差分のフィルタを入力音声波形に適用する
    b = np.apply_along_axis(sptk.mc2b, 1, diff_mgc, alpha)
    synthesizer = Synthesizer(MLSADF(order=dim-1, alpha=alpha), 80)
    owav = synthesizer.synthesis(data, b)

    owav = np.clip(owav, -32768, 32767)
    wavfile.write(outfile, fs, owav.astype(np.int16))
