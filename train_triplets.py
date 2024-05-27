# from relgraph import generate_relation_triplets
from dataset import TrainData, TestNewData
from tqdm import tqdm
import random
from model import InGram
import torch
import numpy as np
from utils import generate_neg, initialize, print_metrics, get_metrics
import os
from evaluation import evaluate
from my_parser import parse
import gc

OMP_NUM_THREADS = 8
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)
torch.cuda.empty_cache()

args = parse()

assert args.data_name in os.listdir(args.data_path), f"{args.data_name} Not Found"
path = args.data_path + args.data_name + "/"
train = TrainData(path, qual=False, special_relation=args.special_relation)
valid = TestNewData(path, qual=False, data_type="valid", special_relation=args.special_relation) 

if not args.no_write:
	os.makedirs(f"./ckpt/{args.exp}/{args.data_name}", exist_ok=True)
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
epochs = args.num_epoch
valid_epochs = args.validation_epoch
num_neg = args.num_neg

ingram_trip = InGram(dim_ent = d_e, hid_dim_ratio_ent = hdr_e, dim_rel = d_r, hid_dim_ratio_rel = hdr_r, \
				num_bin = B, num_layer_ent = args.num_layer_ent, num_layer_rel = args.num_layer_rel, \
				num_head = args.num_head)
ingram_trip = ingram_trip.cuda()
loss_fn = torch.nn.MarginRankingLoss(margin = args.margin, reduction = 'mean')

optimizer = torch.optim.Adam(ingram_trip.parameters(), lr = args.learning_rate)
pbar = tqdm(range(epochs))

losses = []
best_mrr = 0

for epoch in pbar:
	ingram_trip.train()
	optimizer.zero_grad()
	msg, sup = train.split_transductive(0.75)

	init_emb_ent, init_emb_rel, relation_triplets = initialize(train, msg, d_e, d_r, B)
	msg = torch.tensor(msg).cuda()
	sup = torch.tensor(sup).cuda()

	emb_ent, emb_rel = ingram_trip(init_emb_ent, init_emb_rel, msg, relation_triplets)
	pos_scores = ingram_trip.score(emb_ent, emb_rel, sup)
	neg_scores = ingram_trip.score(emb_ent, emb_rel, generate_neg(sup, train.num_ent, num_neg = num_neg))

	loss = loss_fn(pos_scores.repeat(num_neg), neg_scores, torch.ones_like(neg_scores))

	loss.backward()
	torch.nn.utils.clip_grad_norm_(ingram_trip.parameters(), 0.1, error_if_nonfinite = False)
	optimizer.step()
	losses.append(loss.item())
	pbar.set_description(f"loss {loss.item()}")	

	if ((epoch + 1) % valid_epochs) == 0:
		print("Validation")
		ingram_trip.eval()
		val_init_emb_ent, val_init_emb_rel, val_relation_triplets = initialize(valid, valid.msg_triplets, \
																				d_e, d_r, B)

		ranks = evaluate(ingram_trip, valid, val_init_emb_ent, val_init_emb_rel, val_relation_triplets)
		print_metrics(f"Validation Triplets Epoch {epoch + 1}", ranks)
		_, mrr, _, _, _ = get_metrics(ranks)

		if mrr > best_mrr:
			torch.save({'model_state_dict': ingram_trip.state_dict(), \
						'optimizer_state_dict': optimizer.state_dict(), \
						'inf_emb_ent': val_init_emb_ent, \
						'inf_emb_rel': val_init_emb_rel}, \
				f"ckpt/{args.exp}/{args.data_name}/{file_format}_best_triplet.ckpt")
			best_mrr = mrr
		
del ingram_trip, msg, sup
_ = gc.collect()
torch.cuda.empty_cache()

print("Finished Training")
print(best_mrr)
losses = np.array(losses)
np.save(f"ckpt/{args.exp}/{args.data_name}/{file_format}_loss_triplet.npy", losses)