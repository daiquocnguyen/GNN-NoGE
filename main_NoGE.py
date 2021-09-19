from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
import argparse
import scipy.sparse as sp
from utils_NoGE import *
from model_NoGE import *

torch.manual_seed(1337)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)
np.random.seed(1337)

class NoGE:
    """ Node Coherence-based Graph Neural Networks for Knowledge Graph Link Prediction """
    def __init__(self, encoder="QGNN", decoder="QuatE", num_iterations=4000, batch_size=1024, learning_rate=0.01, label_smoothing=0.1,
                 hidden_dim=128, emb_dim=128, num_layers=1, variant="N", eval_step=1, eval_after=1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.label_smoothing = label_smoothing
        self.eval_step = eval_step
        self.eval_after = eval_after
        self.encoder = encoder
        self.decoder = decoder
        self.hid_dim = hidden_dim
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.variant = variant

    """ Functions are adapted from https://github.com/ibalazevic/TuckER for using 1-N scoring strategy """
    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        return np.array(batch), targets.to(device)

    # evaluation
    def evaluate(self, model, data, lst_indexes):
        model.eval()
        with torch.no_grad():
            hits = []
            ranks = []
            for i in range(10):
                hits.append([])

            test_data_idxs = self.get_data_idxs(data)
            er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))
            print("Number of data points: %d" % len(test_data_idxs))

            for i in range(0, len(test_data_idxs), self.batch_size):
                data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
                e1_idx = torch.tensor(data_batch[:, 0]).to(device)
                r_idx = torch.tensor(data_batch[:, 1]).to(device)
                e2_idx = torch.tensor(data_batch[:, 2]).to(device)

                predictions = model.forward(e1_idx, r_idx, lst_indexes).detach()

                for j in range(data_batch.shape[0]):
                    filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                    target_value = predictions[j, e2_idx[j]].item()
                    predictions[j, filt] = 0.0
                    predictions[j, e2_idx[j]] = target_value

                sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

                sort_idxs = sort_idxs.cpu().numpy()
                for j in range(data_batch.shape[0]):
                    rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                    ranks.append(rank + 1)
                    for hits_level in range(10):
                        if rank <= hits_level:
                            hits[hits_level].append(1.0)
                        else:
                            hits[hits_level].append(0.0)

        print('Hits @10: {0}'.format(np.mean(hits[9]) * 100))
        print('Hits @3: {0}'.format(np.mean(hits[2]) * 100))
        print('Hits @1: {0}'.format(np.mean(hits[0]) * 100))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

        return np.mean(hits[9]) * 100, np.mean(hits[2]) * 100, np.mean(hits[0]) * 100, np.mean(ranks), np.mean(1. / np.array(ranks))

    # training and evaluating
    def train_and_eval(self):
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}
        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        print("Creating the new weighted Adj matrix!")
        adj = compute_weighted_adj_matrix(d.train_data, self.entity_idxs, self.relation_idxs).to(device)

        if self.encoder.lower() == "gcn":
            print("Training with the GCN encoder")
            model = NoGE_GCN_QuatE(self.emb_dim, self.hid_dim, adj, len(self.entity_idxs), len(self.relation_idxs), self.num_layers).to(device)
            print("and the customized QuatE decoder...")

        else:
            print("Training with the QGNN encoder")
            if self.decoder.lower() == "quate":
                model = NoGE_QGNN_QuatE(self.emb_dim, self.hid_dim, adj, len(self.entity_idxs), len(self.relation_idxs), self.num_layers, self.variant).to(device)
                print("and the customized QuatE decoder...")
            else:
                model = NoGE_QGNN_DistMult(self.emb_dim, self.hid_dim, adj, len(self.entity_idxs), len(self.relation_idxs), self.num_layers).to(device)
                print("and the customized DistMult decoder...")

        print("Using Adam optimizer")
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        lst_indexes = torch.LongTensor([i for i in range(len(d.entities) + len(d.relations))]).to(device)
        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())
        max_valid = 0.0
        final_test_h10 = 0.0
        final_test_h3 = 0.0
        final_test_h1 = 0.0
        final_test_mr = 0.0
        final_test_mrr = 0.0
        best_epoch = 0
        print("Starting training...")
        for it in range(1, self.num_iterations + 1):
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:, 0]).to(device)
                r_idx = torch.tensor(data_batch[:, 1]).to(device)

                predictions = model.forward(e1_idx, r_idx, lst_indexes)
                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))
                loss = model.loss(predictions, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # prevent the exploding gradient problem
                opt.step()
                losses.append(loss.item())
            print("Epoch: {}".format(it), " --> Loss: {:.4f}".format(np.sum(losses)))
            # evaluation
            if it > self.eval_after and it % self.eval_step == 0:
                print("Validation:")
                tmp_hit10, _, _, _, tmp_mrr = self.evaluate(model, d.valid_data, lst_indexes)
                if max_valid < tmp_mrr:
                    max_valid = tmp_mrr
                    best_epoch = it
                    print("Test:")
                    final_test_h10, final_test_h3, final_test_h1, final_test_mr, final_test_mrr = self.evaluate(model, d.test_data, lst_indexes)

                print("Best valid epoch", best_epoch, " --> Final test results: ", final_test_h10, final_test_h3, final_test_h1, final_test_mr, final_test_mrr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="codex-s", nargs="?", help="codex-s, codex-m, and codex-l")
    parser.add_argument("--num_iterations", type=int, default=3000, nargs="?", help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=1024, nargs="?", help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.005, nargs="?", help="Learning rate.")
    parser.add_argument("--hidden_dim", type=int, default=256, nargs="?", help="")
    parser.add_argument("--emb_dim", type=int, default=256, nargs="?", help="")
    parser.add_argument("--num_layers", type=int, default=1, nargs="?", help="Number of layers")
    parser.add_argument("--encoder", type=str, default="QGNN", nargs="?")
    parser.add_argument("--decoder", type=str, default="QuatE", nargs="?")
    parser.add_argument("--variant", type=str, default="D", nargs="?", help="N: QGNN, D: Dual QGNN")
    parser.add_argument("--eval_step", type=int, default=1, nargs="?")
    parser.add_argument("--eval_after", type=int, default=1000, nargs="?")
    args = parser.parse_args()

    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True
    d = Data(data_dir=data_dir)

    gnnkge = NoGE(encoder=args.encoder, decoder=args.decoder, num_iterations=args.num_iterations, batch_size=args.batch_size,
                learning_rate=args.lr, hidden_dim=args.hidden_dim, emb_dim=args.emb_dim, num_layers=args.num_layers,
                eval_step=args.eval_step, eval_after=args.eval_after, variant=args.variant)

    gnnkge.train_and_eval()
