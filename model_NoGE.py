import numpy as np
import torch
from torch.nn.init import xavier_normal_
from torch import empty, matmul, tensor
import torch
from torch.cuda import empty_cache
from torch.nn import Parameter, Module
from torch.nn.functional import normalize
from tqdm.autonotebook import tqdm
import torch.nn.functional as F
import math
import numpy as np
from gnn_layers import *


'''@Dai Quoc Nguyen'''
''' QGNN encoder - customized DistMult decoder '''
class NoGE_QGNN_DistMult(torch.nn.Module):
    def __init__(self, emb_dim, hid_dim, adj, n_entities, n_relations, num_layers=1):
        super(NoGE_QGNN_DistMult, self).__init__()

        self.adj = adj
        self.num_layers = num_layers
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, emb_dim)
        torch.nn.init.xavier_normal_(self.embeddings.weight.data)

        self.lst_qgnn = torch.nn.ModuleList()
        for _layer in range(self.num_layers):
            if _layer == 0:
                self.lst_qgnn.append(QGNN_layer(emb_dim, hid_dim, act=torch.tanh))
            else:
                self.lst_qgnn.append(QGNN_layer(hid_dim, hid_dim, act=torch.tanh))

        self.bn1 = torch.nn.BatchNorm1d(hid_dim)
        self.hidden_dropout2 = torch.nn.Dropout()
        self.loss = torch.nn.BCELoss()

    def forward(self, e1_idx, r_idx, lst_indexes):
        X = self.embeddings(lst_indexes)
        for _layer in range(self.num_layers):
            X = self.lst_qgnn[_layer](X, self.adj)
        h = X[e1_idx]
        r = X[r_idx + self.n_entities]
        hr = h * r # following the 1-N scoring strategy
        hr = self.bn1(hr)
        hr = self.hidden_dropout2(hr)
        hrt = torch.mm(hr, X[:self.n_entities].t())
        pred = torch.sigmoid(hrt)
        return pred

''' (Dual) QGNN encoder - customized QuatE decoder '''
class NoGE_QGNN_QuatE(torch.nn.Module):
    def __init__(self, emb_dim, hid_dim, adj, n_entities, n_relations, num_layers=1, variant="N"):
        super(NoGE_QGNN_QuatE, self).__init__()
        self.adj = adj
        self.num_layers = num_layers
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, emb_dim)
        torch.nn.init.xavier_normal_(self.embeddings.weight.data)
        self.variant = variant
        self.lst_qgnn = torch.nn.ModuleList()

        qgnn_mode = QGNN_layer
        if self.variant == "D":
            print('Using Dual QGNN!')
            qgnn_mode = DQGNN_layer
        for _layer in range(self.num_layers):
            if _layer == 0:
                self.lst_qgnn.append(qgnn_mode(emb_dim, hid_dim, act=torch.tanh))
            else:
                self.lst_qgnn.append(qgnn_mode(hid_dim, hid_dim, act=torch.tanh))

        self.bn1 = torch.nn.BatchNorm1d(hid_dim)
        self.hidden_dropout2 = torch.nn.Dropout()
        self.loss = torch.nn.BCELoss()

    def forward(self, e1_idx, r_idx, lst_indexes):
        X = self.embeddings(lst_indexes)
        for _layer in range(self.num_layers):
            X = self.lst_qgnn[_layer](X, self.adj)
        h = X[e1_idx]
        r = X[r_idx + self.n_entities]
        T = X[:self.n_entities]
        if self.variant == "D":
            # normalized_r = dual_normalization(r)
            # hr = vec_vec_dual_multiplication(h, normalized_r)
            size = h.size(1) // 8
            hr1, hi1, hj1, hk1, hr2, hi2, hj2, hk2 = torch.split(h, size, dim=1)
            h = torch.cat([hr1, hr2, hi1, hi2, hj1, hj2, hk1, hk2], dim=1)

            rr1, ri1, rj1, rk1, rr2, ri2, rj2, rk2 = torch.split(r, size, dim=1)
            r = torch.cat([rr1, rr2, ri1, ri2, rj1, rj2, rk1, rk2], dim=1)

            tr1, ti1, tj1, tk1, tr2, ti2, tj2, tk2 = torch.split(T, size, dim=1)
            T = torch.cat([tr1, tr2, ti1, ti2, tj1, tj2, tk1, tk2], dim=1)

        #else:
        normalized_r = normalization(r)
        hr = vec_vec_wise_multiplication(h, normalized_r)  # following the 1-N scoring strategy

        hr = self.bn1(hr)
        hr = self.hidden_dropout2(hr)
        hrt = torch.mm(hr, T.t())
        pred = torch.sigmoid(hrt)
        return pred


''' GCN encoder - customized QuatE decoder '''
class NoGE_GCN_QuatE(torch.nn.Module):
   def __init__(self, emb_dim, hid_dim, adj, n_entities, n_relations, num_layers=1):
       super(NoGE_GCN_QuatE, self).__init__()

       self.adj = adj
       self.num_layers = num_layers
       self.n_entities = n_entities
       self.n_relations = n_relations

       self.embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, emb_dim)
       torch.nn.init.xavier_normal_(self.embeddings.weight.data)

       self.lst_gcn = torch.nn.ModuleList()
       for _layer in range(self.num_layers):
           if _layer == 0:
               self.lst_gcn.append(GraphConvolution(emb_dim, hid_dim, act=torch.tanh))
           else:
               self.lst_gcn.append(GraphConvolution(hid_dim, hid_dim, act=torch.tanh))

       self.bn1 = torch.nn.BatchNorm1d(hid_dim)
       self.hidden_dropout2 = torch.nn.Dropout()
       self.loss = torch.nn.BCELoss()

   def forward(self, e1_idx, r_idx, lst_indexes):
       X = self.embeddings(lst_indexes)
       for _layer in range(self.num_layers):
           X = self.lst_gcn[_layer](X, self.adj)
       h = X[e1_idx]  # I.e., simply splitting an embedding into 4 quaternion components, slightly better than using X=np.tile(X, 4) in preliminary experiments. Note that when using X=np.tile(X, 4), QuatE becomes DistMult
       r = X[r_idx + self.n_entities]
       normalized_r = normalization(r)
       hr = vec_vec_wise_multiplication(h, normalized_r) # following the 1-N scoring strategy
       hr = self.bn1(hr)
       hr = self.hidden_dropout2(hr)
       hrt = torch.mm(hr, X[:self.n_entities].t())
       pred = torch.sigmoid(hrt)
       return pred

#Dual quaternion
def dual_normalization(dual_q, split_dim=1): # bs x 8dim; 4xdim for each quaternion part
    '''normalization(a, b) = normalization(a) + e x ( b / norm(a) - a x (a_rb_r + a_ib_i + a_jb_j + a_kb_k) / norm(a) / norm(a)^2)
                            = normalization(a) + e x (b / norm(a) - normalization(a) x (a_rb_r + a_ib_i + a_jb_j + a_kb_k) / norm(a)^2 ) '''
    if len(dual_q.size()) == 1:
        dual_q = dual_q.unsqueeze(0)
    size = dual_q.size(1) // 2
    a, b = torch.split(dual_q, [size, size], dim=1)
    normalized_a, norm_a, q_a, dim = normalization_v2(a, split_dim) # bs x 4 x dim, bs x 1 x dim

    q_b = b.reshape(-1, 4, dim) # bs x 4 x dim
    b_div_norm_a = q_b / norm_a
    inner_ab = torch.sum(q_a * q_b, 1, True)
    normalization_a_time_inner_ab_div_norm_a_2 = normalized_a * inner_ab / (norm_a**2)
    out_b = b_div_norm_a - normalization_a_time_inner_ab_div_norm_a_2

    return torch.cat([normalized_a.reshape(-1, 4 * dim), out_b.reshape(-1, 4 * dim)], dim=1)
#
def vec_vec_dual_multiplication(q, p):
    '''(a,b)*(c,d)=(a*c, a*d+b*c). * denotes the Hamilton product'''
    size = q.size(1) // 2
    a, b = torch.split(q, [size, size], dim=1)
    c, d = torch.split(p, [size, size], dim=1)
    ac = vec_vec_wise_multiplication(a, c)
    ad = vec_vec_wise_multiplication(a, d)
    bc = vec_vec_wise_multiplication(b, c)
    ad_plus_bc = ad + bc

    return torch.cat([ac, ad_plus_bc], dim=1)

# Quaternion operations
def normalization(quaternion, split_dim=1):  # vectorized quaternion bs x 4dim
    size = quaternion.size(split_dim) // 4
    quaternion = quaternion.reshape(-1, 4, size)  # bs x 4 x dim
    quaternion = quaternion / torch.sqrt(torch.sum(quaternion ** 2, 1, True))  # quaternion / norm
    quaternion = quaternion.reshape(-1, 4 * size)
    return quaternion

def normalization_v2(quaternion, split_dim=1):  # vectorized quaternion bs x 4dim
    size = quaternion.size(split_dim) // 4
    quaternion = quaternion.reshape(-1, 4, size)  # bs x 4 x dim
    norm_q = torch.sqrt(torch.sum(quaternion ** 2, 1, True)) # bs x 1 x dim
    normalized_q = quaternion / norm_q  # quaternion / norm
    return normalized_q, norm_q, quaternion, size

def make_wise_quaternion(quaternion):  # for vector * vector quaternion element-wise multiplication
    if len(quaternion.size()) == 1:
        quaternion = quaternion.unsqueeze(0)
    size = quaternion.size(1) // 4
    r, i, j, k = torch.split(quaternion, size, dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=1)  # 0, 1, 2, 3 --> bs x 4dim
    i2 = torch.cat([i, r, -k, j], dim=1)  # 1, 0, 3, 2
    j2 = torch.cat([j, k, r, -i], dim=1)  # 2, 3, 0, 1
    k2 = torch.cat([k, -j, i, r], dim=1)  # 3, 2, 1, 0
    return r2, i2, j2, k2

def get_quaternion_wise_mul(quaternion):
    size = quaternion.size(1) // 4
    quaternion = quaternion.view(-1, 4, size)
    quaternion = torch.sum(quaternion, 1)
    return quaternion

def vec_vec_wise_multiplication(q, p):  # vector * vector
    q_r, q_i, q_j, q_k = make_wise_quaternion(q)  # bs x 4dim

    qp_r = get_quaternion_wise_mul(q_r * p)  # qrpr−qipi−qjpj−qkpk
    qp_i = get_quaternion_wise_mul(q_i * p)  # qipr+qrpi−qkpj+qjpk
    qp_j = get_quaternion_wise_mul(q_j * p)  # qjpr+qkpi+qrpj−qipk
    qp_k = get_quaternion_wise_mul(q_k * p)  # qkpr−qjpi+qipj+qrpk

    return torch.cat([qp_r, qp_i, qp_j, qp_k], dim=1)

def regularization(quaternion):  # vectorized quaternion bs x 4dim
    size = quaternion.size(1) // 4
    r, i, j, k = torch.split(quaternion, size, dim=1)
    return torch.mean(r ** 2) + torch.mean(i ** 2) + torch.mean(j ** 2) + torch.mean(k ** 2)

"Dual QuatE"
class DQuatE(torch.nn.Module):
    def __init__(self, emb_dim, n_entities, n_relations):
        super(DQuatE, self).__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, emb_dim)
        torch.nn.init.xavier_normal_(self.embeddings.weight.data)
        self.bn1 = torch.nn.BatchNorm1d(emb_dim)
        self.hidden_dropout2 = torch.nn.Dropout()
        self.loss = torch.nn.BCELoss()
    def forward(self, e1_idx, r_idx, lst_indexes):
        X = self.embeddings(lst_indexes)
        h = X[e1_idx]
        r = X[r_idx + self.n_entities]
        normalized_r = dual_normalization(r)
        hr = vec_vec_dual_multiplication(h, normalized_r)
        hr = self.bn1(hr)
        hr = self.hidden_dropout2(hr)
        hrt = torch.mm(hr, X[:self.n_entities].t())
        pred = torch.sigmoid(hrt)
        return pred


''' The re-implementation of Quaternion Knowledge Graph Embeddings (https://arxiv.org/abs/1904.10281), following the 1-N scoring strategy '''
class QuatE(torch.nn.Module):
    def __init__(self, emb_dim, n_entities, n_relations):
        super(QuatE, self).__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, emb_dim)
        torch.nn.init.xavier_normal_(self.embeddings.weight.data)
        self.loss = torch.nn.BCELoss()
    def forward(self, e1_idx, r_idx, lst_indexes):
        X = self.embeddings(lst_indexes)
        h = X[e1_idx]
        r = X[r_idx + self.n_entities]
        normalized_r = normalization(r)
        hr = vec_vec_wise_multiplication(h, normalized_r)
        hrt = torch.mm(hr, X[:self.n_entities].t())  # following the 1-N scoring strategy in ConvE
        pred = torch.sigmoid(hrt)
        return pred


''' DistMult, following the 1-N scoring strategy '''
class DistMult(torch.nn.Module):
    def __init__(self, emb_dim, n_entities, n_relations):
        super(DistMult, self).__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, emb_dim)
        torch.nn.init.xavier_normal_(self.embeddings.weight.data)
        self.loss = torch.nn.BCELoss()

    def forward(self, e1_idx, r_idx, lst_indexes):
        X = self.embeddings(lst_indexes)
        h = X[e1_idx]
        r = X[r_idx + self.n_entities]
        hr = h * r
        hrt = torch.mm(hr, X[:self.n_entities].t())  # following the 1-N scoring strategy in ConvE
        pred = torch.sigmoid(hrt)
        return pred


