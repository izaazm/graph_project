# Generate new knowledge graph of test and inference (kg_inference)
from utils import *
import argparse
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('data_src')
parser.add_argument('data_tgt')
parser.add_argument('n_train', type=int)
parser.add_argument('n_test', type=int)
parser.add_argument('p_rel', type=float)
parser.add_argument('p_tri', type=float)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--no_save', default=False, action='store_true')
args = parser.parse_args()

seed = int(100 * args.n_train * args.n_test / args.p_rel * args.seed)

random.seed(seed)

### Read entities/relations ###
_, _, _, fact_all = read_HKG(f"{args.data_src}.json")

### Take GCC ###
gcc_all = gcc(fact_all)
entity, relation, triplet, fact = [], [], [], []
for (h, r, t) in fact_all:
	if h in gcc_all:
		entity.append(h)
		entity.append(t)
		relation.append(r)
		triplet.append((h, r, t))

		qual = []
		for (q, v) in fact_all[(h, r, t)]:
			if v in gcc_all:
				entity.append(v)
				relation.append(q)
				qual.append((q, v))
		
		fact.append(((h, r, t), qual)) 

entity = remove_duplicate(entity)
relation = remove_duplicate(relation)
facts = remove_duplicate_facts(fact)

### Split relation set into train/valid/test ###
num_relation = len(relation)
random.shuffle(relation)
relation_test = relation[:int(num_relation * args.p_rel)]
relation_train = relation[int(num_relation * args.p_rel):]

relation_test = set(relation_test)
relation_train = set(relation_train)

### Sample neighbors from train seeds ###
seed_train = random.sample(entity, args.n_train)
entity_train = sample_2hop(facts, seed_train, 50)

### Generate train set ###
train_all = dict()
for h, r, t in triplet:
	if h in entity_train and r in relation_train and t in entity_train:
		qual = []
		for (q, v) in facts[(h, r, t)]:
			if v in entity_train and q in relation_train:
				qual.append((q, v))
		
		train_all[(h, r, t)] = qual

### Take GCC ###
gcc_train = gcc(train_all)
train = dict()
for h, r, t in train_all:
	if h in gcc_train:
		train[(h, r, t)] = train_all[(h, r, t)]

### Remove train entities ###
triplet_p = []
for h, r, t in triplet:
	if h not in gcc_train and t not in gcc_train:
		triplet_p.append((h, r, t))
entity_p, relation_p = gather(triplet_p)

### Sample neighbors from valid seeds ###
seed_test = random.sample(entity_p, args.n_test)
entity_test = sample_2hop(triplet_p, seed_test, 50)

### Generate test set ###
test_x = []
test_y = []
for h, r, t in triplet_p:
	if h in entity_test and r in relation_train and t in entity_test:
		qual = []
		for (q, v) in facts[(h, r, t)]:
			if v in entity_test and q in relation_train:
				qual.append((q, v))
		test_x.append(((h, r, t), qual))
	elif h in entity_test and r in relation_test and t in entity_test:
		qual = []
		for (q, v) in facts[(h, r, t)]:
			if v in entity_test and q in relation_test:
				qual.append((q, v))
		test_y.append(((h, r, t), qual))

### Merge X_test and Y_test ###
test_all = merge(test_x, test_y, args.p_tri)
test_all = dict(test_all)

### Take GCC ###
gcc_test = gcc(test_all)
test = dict()
for h, r, t in test_all:
	if h in gcc_test:
		test[(h, r, t)] = test_all[(h, r, t)]

### Check no overlap ###
check_no_overlap(gcc_train, gcc_test)

print("Number of triplets in Train KG:", len(train))
print("Number of triplets in Inference KG:", len(test))

### Save files ###
if not args.no_save:
	save_dir = f"./{args.data_tgt}/"
	os.makedirs(save_dir, exist_ok=True)
	write(save_dir + 'train.json', train)
	write(save_dir + 'kg_inference.json', test) 