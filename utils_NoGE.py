from load_data import Data
import numpy as np
import time
import torch
from collections import defaultdict
import argparse
import scipy.sparse as sp
from collections import Counter
import itertools
from scipy import sparse

torch.manual_seed(1337)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)
np.random.seed(1337)

def normalize_sparse(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# The new weighted Adj matrix
def compute_weighted_adj_matrix(data, entity_idxs, relation_idxs):
    num_entities = len(entity_idxs)
    tok2indx = dict()
    for _key in entity_idxs.keys():
        tok2indx[_key] = entity_idxs[_key]
    for _key in relation_idxs.keys():
        tok2indx[_key] = relation_idxs[_key] + num_entities

    # Skipgrams
    back_window = 2
    front_window = 2
    skipgram_counts = Counter()
    for iheadline, headline in enumerate(data):
        tokens = [tok2indx[tok] for tok in headline]
        for ii_word, word in enumerate(tokens):
            ii_context_min = max(0, ii_word - back_window)
            ii_context_max = min(len(headline) - 1, ii_word + front_window)
            ii_contexts = [
                ii for ii in range(ii_context_min, ii_context_max + 1)
                if ii != ii_word]
            for ii_context in ii_contexts:
                skipgram = (tokens[ii_word], tokens[ii_context])
                skipgram_counts[skipgram] += 1

    # Word-Word Count Matrix
    row_indxs = []
    col_indxs = []
    dat_values = []
    for (tok1, tok2), sg_count in skipgram_counts.items():
        row_indxs.append(tok1)
        col_indxs.append(tok2)
        dat_values.append(sg_count)
    wwcnt_mat = sparse.csr_matrix((dat_values, (row_indxs, col_indxs)))
    num_skipgrams = wwcnt_mat.sum()
    assert (sum(skipgram_counts.values()) == num_skipgrams)

    # for creating sparse matrices
    row_indxs = []
    col_indxs = []
    weighted_edges = []
    # reusable quantities
    sum_over_contexts = np.array(wwcnt_mat.sum(axis=1)).flatten()
    # computing weights for edges
    for (tok_word, tok_context), sg_count in skipgram_counts.items():
        nwc = sg_count
        Pwc = nwc / num_skipgrams
        nw = sum_over_contexts[tok_word]
        Pw = nw / num_skipgrams
        #
        edge_val = Pwc / Pw # for entity-entity edges
        if tok_word > len(entity_idxs) or tok_context > len(entity_idxs): # for relation-entity edges
            edge_val = Pwc
        row_indxs.append(tok_word)
        col_indxs.append(tok_context)
        weighted_edges.append(edge_val)
    edge_mat = sparse.csr_matrix((weighted_edges, (row_indxs, col_indxs)))
    # adding self-loop
    adj = edge_mat + sparse.eye(edge_mat.shape[0], format="csr")
    # print(adj)
    adj = normalize_sparse(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

