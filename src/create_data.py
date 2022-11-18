import scipy.io.wavfile as wavfile
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description = "Make a binary derivative data from wav file")
parser.add_argument("-f", "--file", metavar="WAV", help = "Path to your input wav file")
args = vars(parser.parse_args())


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
            stream.append(plus)
            cur += threshold
        elif dif < 0:
            stream.append(minus)
            cur -= threshold
        else:
            print("hoge")

    return bytes(stream)


if not os.path.exists("data/binary"):
    os.mkdir("data/binary")

wf = args["file"]
print("file: {}".format(wf))
fs, data = wavfile.read("data/wav/{}".format(wf))
data = data.astype(float)
scale = fs//SR

stream = encode_dm(source=data, scale=scale, threshold=threshold)
bn, _ = os.path.splitext(wf)
with open("data/binary/{}_SR{}_threshold{}".format(bn, SR, threshold),"wb") as f:
    f.write(stream)
