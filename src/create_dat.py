import os
import sys
import array

from fastdtw import fastdtw
import numpy as np
import pysptk as sptk

srcspk = os.environ.get('SRC', 'src')
tgtspk = os.environ.get('TRG', 'trg')

mgclist = os.listdir("data/mgc/{}".format(srcspk))

if not os.path.isdir("data/dat"):
    os.mkdir("data/dat")
if not os.path.isdir("data/dat/{}".format(srcspk)):
    os.mkdir("data/dat/{}".format(srcspk))
if not os.path.isdir("data/dat/{}".format(tgtspk)):
    os.mkdir("data/dat/{}".format(tgtspk))

def distfunc(x,y):
# Euclid distance except first dim
    return np.linalg.norm(x[1:]-y[1:])

dim = 25 # mgc dim + 1
for mf in mgclist:
    print(mf)
    bn, _ = os.path.splitext(mf)
    srcfile = "data/mgc/{}/{}".format(srcspk,mf)
    tgtfile = "data/mgc/{}/{}".format(tgtspk,mf)

    with open(srcfile,"rb") as f:
        x = np.fromfile(f, dtype="<f8", sep="")
        x = x.reshape(len(x)//dim,dim)
    with open(tgtfile,"rb") as f:
        y = np.fromfile(f, dtype="<f8", sep="")
        y = y.reshape(len(y)//dim,dim)
    print("framelen: (x,y) = {} {}".format(len(x),len(y)))
    _,twf = fastdtw(x,y,dist=distfunc)
    srcout = "data/dat/{}/{}.dat".format(srcspk,bn)
    tgtout = "data/dat/{}/{}.dat".format(tgtspk,bn)
    twfx = list(map(lambda l: l[0], twf))
    twfy = list(map(lambda l: l[1], twf))

    with open(srcout,"wb") as f:
        x[twfx].tofile(f)
    with open(tgtout,"wb") as f:
        y[twfy].tofile(f)
