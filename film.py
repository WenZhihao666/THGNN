import torch
from torch import nn
from torch.nn import functional as F
import sys

class Scale_4(nn.Module):
    def __init__(self, args):
        super(Scale_4, self).__init__()
        self.vars = nn.ParameterList()
        self.args = args
        self.w1 = nn.Parameter(torch.ones(*[args.out_dim + 1, args.out_dim], requires_grad=True))
        torch.nn.init.kaiming_normal_(self.w1)
        self.b1 = nn.Parameter(torch.zeros(args.out_dim + 1, requires_grad=True))

    def forward(self, x):
        x = F.linear(x, self.w1, self.b1)
        # x = torch.relu(x)
        x = F.leaky_relu(x)

        x = x.T
        x1 = x[:self.args.out_dim].T #.view(x.size(0), self.args.out_dim)
        x2 = x[self.args.out_dim:].T #.view(x.size(0), 1)
        para_list = [x1, x2]
        # print(x1.shape)
        # print(x2.shape)
        return para_list

class Shift_4(nn.Module):
    def __init__(self, args):
        super(Shift_4, self).__init__()
        self.args = args
        self.vars = nn.ParameterList()
        self.w1 = nn.Parameter(torch.ones(*[args.out_dim + 1, args.out_dim], requires_grad=True))
        torch.nn.init.kaiming_normal_(self.w1)
        self.b1 = nn.Parameter(torch.zeros(args.out_dim + 1, requires_grad=True))

    def forward(self, x):
        x = F.linear(x, self.w1, self.b1)
        # x = torch.relu(x)
        x = F.leaky_relu(x)
        # x = torch.squeeze(x)
        x = x.T
        x1 = x[:self.args.out_dim].T #.view(x.size(0), self.args.out_dim)
        x2 = x[self.args.out_dim:].T #.view(x.size(0), 1)
        para_list = [x1, x2]
        return para_list

