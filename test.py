from relgraph import generate_relation_triplets
from dataset import TestNewData
from tqdm import tqdm
import random
from model import InGram
import torch
import numpy as np
from utils import get_rank, get_metrics, print_metrics
from my_parser import parse
from evaluation import evaluate
import os

OMP_NUM_THREADS=8
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)
torch.cuda.empty_cache()

args = parse(test=True)

assert args.data_name in os.listdir(args.data_path), f"{args.data_name} Not Found"

path = args.data_path + args.data_name + "/"
test_triplet = TestNewData(path, data_type = "test", qual = False, special_relation=args.special_relation)
test_qual = TestNewData(path, data_type = "test", qual = True, special_relation=args.special_relation)

if not args.best:
	file_format = f"lr_{args.learning_rate}_dim_{args.dimension_entity}_{args.dimension_relation}" + \
				f"_bin_{args.num_bin}_total_{args.num_epoch}_every_{args.validation_epoch}" + \
				f"_neg_{args.num_neg}_layer_{args.num_layer_ent}_{args.num_layer_rel}" + \
				f"_hid_{args.hidden_dimension_ratio_entity}_{args.hidden_dimension_ratio_relation}" + \
				f"_head_{args.num_head}_margin_{args.margin}"

d_e = args.dimension_entity
d_r = args.dimension_relation
hdr_e = args.hidden_dimension_ratio_entity
hdr_r = args.hidden_dimension_ratio_relation
B = args.num_bin
num_neg = args.num_neg

InGram_triplet = InGram(dim_ent = d_e, hid_dim_ratio_ent = hdr_e, dim_rel = d_r, hid_dim_ratio_rel = hdr_r,\
						num_bin = B, num_layer_ent = args.num_layer_ent, num_layer_rel = args.num_layer_rel, \
						num_head = args.num_head)
InGram_qual = InGram(dim_ent = d_e, hid_dim_ratio_ent = hdr_e, dim_rel = d_r, hid_dim_ratio_rel = hdr_r,\
					 num_bin = B, num_layer_ent = args.num_layer_ent, num_layer_rel = args.num_layer_rel, \
					 num_head = args.num_head)
InGram_triplet = InGram_triplet.cuda()
InGram_qual = InGram_qual.cuda()

InGram_triplet.load_state_dict(torch.load(f"ckpt/{args.exp}/{args.data_name}/{file_format}_best_triplet.ckpt")["model_state_dict"])
InGram_qual.load_state_dict(torch.load(f"ckpt/{args.exp}/{args.data_name}/{file_format}_best_qual.ckpt")["model_state_dict"])

print("Test")
print()
InGram_triplet.eval()
InGram_qual.eval()

test_msg_triplet = test_triplet.msg_triplets
test_sup_triplet = test_triplet.sup_triplets
test_triplet_relation_triplets = generate_relation_triplets(test_msg_triplet, test_triplet.num_ent, test_triplet.num_rel, B)

test_msg_qual = test_triplet.msg_triplets
test_sup_qual = test_triplet.sup_triplets
test_qual_relation_triplets = generate_relation_triplets(test_msg_qual, test_qual.num_ent, test_qual.num_rel, B)

test_triplet_init_emb_ent = torch.load(f"ckpt/{args.exp}/{args.data_name}/{file_format}_best_triplet.ckpt")["inf_emb_ent"]
test_triplet_init_emb_rel = torch.load(f"ckpt/{args.exp}/{args.data_name}/{file_format}_best_triplet.ckpt")["inf_emb_rel"]

test_qual_init_emb_ent = torch.load(f"ckpt/{args.exp}/{args.data_name}/{file_format}_best_qual.ckpt")["inf_emb_ent"]
test_qual_init_emb_rel = torch.load(f"ckpt/{args.exp}/{args.data_name}/{file_format}_best_qual.ckpt")["inf_emb_rel"]

test_triplet_sup = torch.tensor(test_sup_triplet).cuda()
test_triplet_msg = torch.tensor(test_msg_triplet).cuda()

test_qual_sup = torch.tensor(test_sup_qual).cuda()
test_qual_msg = torch.tensor(test_msg_qual).cuda()

test_triplet_relation_triplets = torch.tensor(test_triplet_relation_triplets).cuda()
test_qual_relation_triplets = torch.tensor(test_qual_relation_triplets).cuda()

ranks_triplet = evaluate(InGram_triplet, test_triplet, test_triplet_init_emb_ent, test_triplet_init_emb_rel, test_triplet_relation_triplets)
ranks_qual = evaluate(InGram_qual, test_qual, test_qual_init_emb_ent, test_qual_init_emb_rel, test_qual_relation_triplets)
ranks = ranks_triplet + ranks_qual

print_metrics("Test", ranks)
