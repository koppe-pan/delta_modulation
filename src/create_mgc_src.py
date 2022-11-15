import scipy.io.wavfile as wavfile
import pyworld as pw
import pysptk as sptk
import numpy as np
import os

SR = int(os.environ.get('SR', '16000'))
threshold = int(os.environ.get('THRESHOLD', '1000'))
minus = os.environ.get('MINUS', '0').lower() in ('true', '1', 't')
plus = os.environ.get('PLUS', '1').lower() in ('true', '1', 't')

def encode_dm(source, scale, threshold):
    stream = []
    L = (len(source)+scale-1) // scale
    cur = 0
    for i in range(L-1):
        dif = source[i*scale] - cur
        if dif >= 0:
            cur += threshold
        elif dif < 0:
            cur -= threshold
        else:
            print("hoge")
        stream.append(cur)

    return stream


if not os.path.exists("data/mgc"):
    os.mkdir("data/mgc")

if not os.path.exists("data/mgc/src"):
    os.mkdir("data/mgc/src")

wavlist = os.listdir("data/wav")
for wf in wavlist:
    print("file: {}".format(wf))
    fs, data = wavfile.read("data/wav/{}".format(wf))
    data = data.astype(float)
    scale = fs//SR

    y = encode_dm(source=data, scale=scale, threshold=threshold)
    data = np.array(y).astype(float)

    f0, t = pw.harvest(data, fs)
    sp = pw.cheaptrick(data, f0, t, fs)
    ap = pw.d4c(data, f0, t, fs)
    alpha = 0.42
    dim = 24
    mgc = sptk.sp2mc(sp, dim, alpha)

    bn, _ = os.path.splitext(wf)

    with open("data/mgc/src/{}.mgc".format(bn),"wb") as f:
        mgc.tofile(f)
