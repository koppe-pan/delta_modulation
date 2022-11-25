import torch
from torch import optim
import os
import numpy as np
from rnn_model import fittingRNN

SR = int(os.environ.get('SR', '16000'))
threshold = int(os.environ.get('THRESHOLD', '1000'))
L = int(os.environ.get('L', '3'))
lr = float(os.environ.get('LR', '0.1'))
epoch_num = int(os.environ.get('EPOCH_NUM', '50'))

c1 = 10000
c2 = 100
c3 = 1

# train
model = fittingRNN()
if not os.path.exists("trained_model"):
    os.mkdir("trained_model")

if os.path.exists("trained_model/fitting_model_rnn.model"):
    model.load_state_dict(torch.load("trained_model/fitting_model_rnn.model"))
loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

dataset = []
wavfile = os.listdir("data/wav")[0:30]
for wf in wavfile:
    bn, _ = os.path.splitext(wf)
    dataset.append(np.load("data/npy/{}_SR{}_threshold{}_L{}.npz".format(bn, SR, threshold, L))['arr_0'])

bits = []
with open("data/binary/{}_SR{}_threshold{}".format(bn, SR, threshold),"rb") as f:
    for b in f.read():
        bits.append(b)

for epoch in range(epoch_num):
    print("{0} / {1} epoch start.".format(epoch + 1, epoch_num))
    sum_loss = 0.0
    for i, data in enumerate(dataset):
        model.reset_state()
        optimizer.zero_grad()
        accum_loss = None
        for j, enum in enumerate(zip(data, data[1:] + [0]*L)):
            cur = [*bits[j:j+L], *enum[0]]
            nex = [*bits[j+1:j+L+1], *enum[1]]
            cur = torch.tensor(cur).to(torch.float32)
            if (len(cur)%L) != 0:
                break
            loss = 0
            x = model(cur)[0]
            y = nex
            for loss_i in range(L-1):
                loss += c1 * (x[loss_i]-y[loss_i])**2
                loss += c2 * (x[L+loss_i]-y[L+loss_i])**2
            loss += c3 * (x[-1]-y[-1])**2
            accum_loss = loss if accum_loss is None else accum_loss + loss
        accum_loss.backward()
        optimizer.step()
        sum_loss += float(accum_loss.data.cpu())
        if (i + 1) % 100 == 0:
            print("{0} / {1} sentences finished.".format(i + 1, len(dataset)))
    print("mean loss = {0}.".format(sum_loss))

    model_file = "trained_model/fitting_model_rnn.model"
    torch.save(model.state_dict(), model_file)
