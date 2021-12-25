<p align="center">
	<img src="https://github.com/daiquocnguyen/GNN-NoGE/blob/master/logo.png" width="250">
</p>

# Node Co-occurrence based Graph Neural Networks for Knowledge Graph Link Prediction<a href="https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2Fdaiquocnguyen%2FGNN-NoGE%2Fblob%2Fmaster%2FREADME.md"><img alt="Twitter" src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fdaiquocng"></a>

<img alt="GitHub top language" src="https://img.shields.io/github/languages/top/daiquocnguyen/GNN-NoGE"><a href="https://github.com/daiquocnguyen/GNN-NoGE/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/daiquocnguyen/GNN-NoGE"></a>
<img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/daiquocnguyen/GNN-NoGE">
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/daiquocnguyen/GNN-NoGE">
<a href="https://github.com/daiquocnguyen/GNN-NoGE/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/daiquocnguyen/GNN-NoGE"></a>
<a href="https://github.com/daiquocnguyen/GNN-NoGE/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/daiquocnguyen/GNN-NoGE"></a>
<img alt="GitHub" src="https://img.shields.io/github/license/daiquocnguyen/GNN-NoGE">

This program provides the implementation of our NoGE as described in [our paper](https://arxiv.org/abs/2104.07396). Given a knowledge graph, NoGE constructs a single graph considering entities and relations as individual nodes and then computes weights for edges among nodes based on the co-occurrence of entities and relations. NoGE leverages vanilla GNNs (such as GCNs, Quaternion GNNs and our proposed Dual Quaternion GNNs) to update vector representations for entity and relation nodes. Then NoGE adopts a score function to produce the triple scores.

<p align="center">
	<img src="https://github.com/daiquocnguyen/GNN-NoGE/blob/master/NoGE.png" width="750">
</p>

## Usage

### Requirements
- Python 3.7
- Pytorch 1.5.0 & CUDA 10.1

### Running commands:

    python -u main_NoGE.py --dataset codex-s --num_iterations 3000 --eval_after 2000 --batch_size 1024 --lr 0.001 --emb_dim 256 --hidden_dim 256 --encoder QGNN --variant D > codexs_dqgnn_256_0001.txt
    
    python -u main_NoGE.py --dataset codex-m --num_iterations 3000 --eval_after 2000 --batch_size 1024 --lr 0.0005 --emb_dim 256 --hidden_dim 256 --encoder QGNN --variant D > codexm_dqgnn_256_00005.txt

    python -u main_NoGE.py --dataset codex-l --num_iterations 1500 --eval_after 1000 --batch_size 1024 --lr 0.0001 --emb_dim 256 --hidden_dim 256 --encoder QGNN --variant D > codexl_dqgnn_256_00001.txt


## Cite 

Please cite the paper whenever NoGE is used to produce published results or incorporated into other software:

    @inproceedings{Nguyen2022NoGE,
        author={Dai Quoc Nguyen and Vinh Tong and Dinh Phung and Dat Quoc Nguyen},
        title={Node Co-occurrence based Graph Neural Networks for Knowledge Graph Link Prediction},
        booktitle={Proceedings of WSDM 2022 (Demonstrations)},
        year={2022}
    }

## License

As a free open-source implementation, NoGE is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed. I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.

NoGE is licensed under the Apache License 2.0.


