import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


def make_quaternion_mul(kernel):
    """" The constructed 'hamilton' W is a modified version of the quaternion representation,
    thus doing tf.matmul(Input,W) is equivalent to W * Inputs. * denotes the Hamilton product."""
    dim = kernel.size(1) // 4
    r, i, j, k = torch.split(kernel, [dim, dim, dim, dim], dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=0)  # 0, 1, 2, 3
    i2 = torch.cat([i, r, -k, j], dim=0)  # 1, 0, 3, 2
    j2 = torch.cat([j, k, r, -i], dim=0)  # 2, 3, 0, 1
    k2 = torch.cat([k, -j, i, r], dim=0)  # 3, 2, 1, 0
    hamilton = torch.cat([r2, i2, j2, k2], dim=1) # Concatenate 4 quaternion components for a faster implementation.
    assert kernel.size(1) == hamilton.size(1)
    return hamilton

def dual_quaternion_mul(A, B, input):
    '''(A, B) * (C, D) = (A * C, A * D + B * C)'''
    dim = input.size(1) // 2
    C, D = torch.split(input, [dim, dim], dim=1)
    A_hamilton = make_quaternion_mul(A)
    B_hamilton = make_quaternion_mul(B)
    AC = torch.mm(C, A_hamilton)
    AD = torch.mm(D, A_hamilton)
    BC = torch.mm(C, B_hamilton)
    AD_plus_BC = AD + BC
    return torch.cat([AC, AD_plus_BC], dim=1)

''' Quaternion graph neural networks! QGNN layer! https://arxiv.org/abs/2008.05089 '''
class QGNN_layer(Module):
    def __init__(self, in_features, out_features, act=torch.tanh):
        super(QGNN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        #
        self.weight = Parameter(torch.FloatTensor(self.in_features // 4, self.out_features))

        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        hamilton = make_quaternion_mul(self.weight)
        support = torch.mm(input, hamilton)  # Hamilton product, quaternion multiplication! Concatenate 4 components of the quaternion input for a faster implementation.
        output = torch.spmm(adj, support)
        output = self.bn(output)
        return self.act(output)


''' Dual quaternion graph neural networks! '''
class DQGNN_layer(Module):
    def __init__(self, in_features, out_features, act=torch.tanh):
        super(DQGNN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        #
        self.A = Parameter(torch.FloatTensor(self.in_features // 8, self.out_features // 2)) # (A, B) = A + eB, e^2 = 0
        self.B = Parameter(torch.FloatTensor(self.in_features // 8, self.out_features // 2))

        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.A.size(0) + self.A.size(1)))
        self.A.data.uniform_(-stdv, stdv)
        self.B.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = dual_quaternion_mul(self.A, self.B, input)
        output = torch.spmm(adj, support)
        output = self.bn(output)
        return self.act(output)

""" Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 """
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, act=torch.relu,  bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.act = act
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        output = self.bn(output)
        return self.act(output)

