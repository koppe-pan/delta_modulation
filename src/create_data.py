import scipy.io.wavfile as wavfile
import numpy as np
import os

minus =  0b0 #BitArray('0b01') # 1
plus = 0b1 #BitArray('0b10') # 2
#zero = 0b00 #BitArray('0b00') # 0

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

wavlist = os.listdir("data/wav")
for wf in wavlist:
    print("file: {}".format(wf))
    fs, data = wavfile.read("data/wav/{}".format(wf))
    data = data.astype(float)

    SR = 16000
    threshold = 1000

    scale = fs//SR

    stream = encode_dm(source=data, scale=scale, threshold=threshold)
    bn, _ = os.path.splitext(wf)

    sample_per_part = 10000
    number_of_parts = len(data)//sample_per_part
    for num in range(number_of_parts):
        with open("data/binary/{}_SR{}_threshold{}_part{}".format(bn, SR, threshold, num),"wb") as f:
            f.write(stream[num*sample_per_part : (num+1)*sample_per_part])
