import torch
import json
import numpy as np
from relgraph import generate_relation_triplets

def remove_duplicate(x):
	return list(dict.fromkeys(x))

def remove_duplicate_facts(facts):
	facts_dict = dict()
	for triplets, qual in facts:
		if triplets in facts_dict:
			facts_dict[triplets].extend(qual)
		else:
			facts_dict[triplets] = qual
	
	for triplets in facts_dict:
		facts_dict[triplets] = remove_duplicate(facts_dict[triplets])

	return facts_dict

def read_HKG(path):
	entity = []
	relation = []
	triplet = []
	facts = []

	with open(path, 'r') as f:
		for line in f.readlines():
			cur_fact = json.loads(line)
			qual = []
			for i, rel in enumerate(cur_fact):
				if i == 0:
					r = rel
					h, t = cur_fact[rel]
				elif rel != "N":
					relation.append(rel)
					qual.append((rel, cur_fact[rel]))

			entity.append(h)
			entity.append(t)
			relation.append(r)
			triplet.append((h, r, t))
			facts.append(((h, r, t), qual))

	return remove_duplicate(entity), remove_duplicate(relation), remove_duplicate(triplet), remove_duplicate_facts(facts)

def initialize(target, msg, d_e, d_r, B):

    init_emb_ent = torch.zeros((target.num_ent, d_e)).cuda()
    init_emb_rel = torch.zeros((2*target.num_rel, d_r)).cuda()
    gain = torch.nn.init.calculate_gain('relu')
    torch.nn.init.xavier_normal_(init_emb_ent, gain = gain)
    torch.nn.init.xavier_normal_(init_emb_rel, gain = gain)
    relation_triplets = generate_relation_triplets(msg, target.num_ent, target.num_rel, B)

    relation_triplets = torch.tensor(relation_triplets).cuda()

    return init_emb_ent, init_emb_rel, relation_triplets

def generate_neg(triplets, num_ent, num_neg = 1):
	neg_triplets = triplets.unsqueeze(dim=1).repeat(1,num_neg,1)
	rand_result = torch.rand((len(triplets),num_neg)).cuda()
	perturb_head = rand_result < 0.5
	perturb_tail = rand_result >= 0.5
	rand_idxs = torch.randint(low=0, high = num_ent-1, size = (len(triplets),num_neg)).cuda()
	rand_idxs[perturb_head] += rand_idxs[perturb_head] >= neg_triplets[:,:,0][perturb_head]
	rand_idxs[perturb_tail] += rand_idxs[perturb_tail] >= neg_triplets[:,:,2][perturb_tail]
	neg_triplets[:,:,0][perturb_head] = rand_idxs[perturb_head]
	neg_triplets[:,:,2][perturb_tail] = rand_idxs[perturb_tail]
	neg_triplets = torch.cat(torch.split(neg_triplets, 1, dim = 1), dim = 0).squeeze(dim = 1)

	return neg_triplets

def get_rank(triplet, scores, filters, target = 0):
	thres = scores[triplet[0,target]].item()
	scores[filters] = thres - 1
	rank = (scores > thres).sum() + (scores == thres).sum()//2 + 1
	return rank.item()

def get_metrics(rank):
	rank = np.array(rank, dtype = np.int)
	mr = np.mean(rank)
	mrr = np.mean(1 / rank)
	hit10 = np.sum(rank < 11) / len(rank)
	hit3 = np.sum(rank < 4) / len(rank)
	hit1 = np.sum(rank < 2) / len(rank)
	return mr, mrr, hit10, hit3, hit1