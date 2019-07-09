import math
import functools
import operator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

def ActorCNNLSTM(nn.Module):
    def __init__(self, action_dim, max_action, args):
        super(ActorLSTM, self).__init__()
        self.args = args
        self.max_action = max_action
        # ONLY TRU IN CASE OF DUCKIETOWN:
        flat_size = 32 * 10 * 15
        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, 32, 7, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1)
        
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.lstm = nn.LSTMCell(flat_size, 256)
        self.lin2 = nn.Linear(256, action_dim)

        if args.spe_init:
            print("use special initializer for actor")
            self.apply(weights_init)
            self.lin2.weight.data = normalized_columns_initializer(
                self.lin2.weight.data, 0.01)
            self.lin2.bias.data.fill_(0)







