import torch
import torch.nn as nn
import os

L = int(os.environ.get('L', '3'))


# model
class fittingRNN(nn.Module):
    def __init__(self, in_neurons = L*2, hidden_neurons = L*4, out_neurons = L*2):
        super(fittingRNN, self).__init__()
        self.hidden_neurons = hidden_neurons
        self.W_x_h = nn.Linear(in_neurons, hidden_neurons)
        self.W_h_h = nn.Linear(hidden_neurons, hidden_neurons)
        self.W_h_y = nn.Linear(hidden_neurons, out_neurons)

        gpu_id = None
        if gpu_id is not None:
            self.device = torch.device(gpu_id)
            self.to(self.device)
        else:
            self.device = torch.device('cpu')

        self.reset_state()

    def reset_state(self):
        self.h = torch.zeros(1, self.hidden_neurons).to(self.device)

    def forward(self, cur):
        self.h = torch.tanh(self.W_x_h(cur.to(self.device)) + self.W_h_h(self.h))
        y = self.W_h_y(self.h)
        return y
