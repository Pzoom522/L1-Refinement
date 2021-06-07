import os
import json
import argparse
import torch
from models import build_model
from trainer import Trainer
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")

parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=10000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")

parser.add_argument("--src_lang", type=str, default='src', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='tgt', help="Target language")
parser.add_argument("--src_emb", type=str, default='', help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default='', help="Reload target embeddings")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")

parser.add_argument("--exp_path", type=str, default="")
parser.add_argument('--log_name', type=str, default="l1_iter_log.txt")
parser.add_argument('--l1_map', type=str, default="best_l1_map.npy")

params = parser.parse_args()

# check parameters
assert params.dico_build in ["S2T", "T2S", "S2T|T2S", "S2T&T2S"]
assert params.dico_max_size == 0 or params.dico_max_size < params.dico_max_rank
assert params.dico_max_size == 0 or params.dico_max_size > params.dico_min_size
assert os.path.isfile(params.src_emb)
assert os.path.isfile(params.tgt_emb)

params.exp_path = params.src_lang + "-" + params.tgt_lang if params.exp_path == "" else params.exp_path


src_emb, tgt_emb, mapping = build_model(params, False)

trainer = Trainer(src_emb, tgt_emb, mapping, None, params)

# get mutual NN dico
trainer.build_dictionary()

trainer.l1_procrustes()

trainer.export()
