from scipy.io import wavfile
import pyworld as pw
import pysptk as sptk
import os

if not os.path.exists("data/mgc"):
    os.mkdir("data/mgc")

if not os.path.exists("data/mgc/trg"):
    os.mkdir("data/mgc/trg")

wavlist = os.listdir("data/wav")
for wf in wavlist:
    print("file: {}".format(wf))
    fs, data = wavfile.read("data/wav/{}".format(wf))
    data = data.astype(float)

    f0, t = pw.harvest(data, fs)
    sp = pw.cheaptrick(data, f0, t, fs)
    ap = pw.d4c(data, f0, t, fs)
    alpha = 0.42
    dim = 24
    mgc = sptk.sp2mc(sp, dim, alpha)

    bn, _ = os.path.splitext(wf)

    with open("data/mgc/trg/{}.mgc".format(bn),"wb") as f:
        mgc.tofile(f)
