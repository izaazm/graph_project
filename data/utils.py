# utility function
import networkx as nx
import random
import json

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
			valid = False
			for i, rel in enumerate(cur_fact):
				if i == 0:
					r = rel
					if len(cur_fact[rel]) == 2:
						h, t = cur_fact[rel]
						valid = True
				elif rel != "N":
					relation.append(rel)
					if isinstance(cur_fact[rel], list):
						qual.append((rel, cur_fact[rel][0]))
					elif isinstance(cur_fact[rel], str):
						qual.append((rel, cur_fact[rel]))
			
			if valid:
				entity.append(h)
				entity.append(t)
				relation.append(r)
				triplet.append((h, r, t))
				facts.append(((h, r, t), qual))

	return remove_duplicate(entity), remove_duplicate(relation), remove_duplicate(triplet), remove_duplicate_facts(facts)

def gather(x):
	ent = []
	rel = []
	for h, r, t in x:
		ent.append(h)
		ent.append(t)
		rel.append(r)
	return remove_duplicate(ent), remove_duplicate(rel)

def check_no_overlap(x, y):
	assert len(set(x).intersection(set(y))) == 0
	print("Done: Check no overlap")

def write(path, facts):
	with open(path, "w") as f:
		for h, r, t in facts:
			n = 2
			fact = dict()
			fact[r] = [h, t]
			for q, v in facts[(h, r, t)]:
				fact[q] = v
				n += 1
			fact["N"] = n
			f.write(json.dumps(fact) + '\n')

def gather_neighbor(triplet, x, thr):
	res = []
	for h, r, t in triplet:
		if h == x:
			res.append(t)
		elif t == x:
			res.append(h)
	res = remove_duplicate(res)
	if len(res) > thr:
		res = random.sample(res, thr)
	return res

def sample_2hop(triplet, x, thr):
	sample = set()
	for e in x:
		neighbor = set([e])
		neighbor_1hop = gather_neighbor(triplet, e, thr)
		neighbor = neighbor.union(set(neighbor_1hop))

		for e1 in neighbor_1hop:
			neighbor_2hop = gather_neighbor(triplet, e1, thr)
			neighbor = neighbor.union(set(neighbor_2hop))

		sample = sample.union(neighbor)
	return sample

def merge(x, y, p):
	if p >= 1:
		return y
	elif p <= 0:
		return x
	else:
		num_tot = min(len(x) / (1 - p), len(y) / p)
		random.shuffle(x)
		random.shuffle(y)
		return x[:int(num_tot * (1 - p))] + y[:int(num_tot * p)]

def gcc(triplet):
	edge = []
	for h, r, t in triplet:
		edge.append((h, t))
	G = nx.Graph()
	G.add_edges_from(edge)
	largest_cc = max(nx.connected_components(G), key=len)
	return largest_cc

def convert_to_qualifier_graph(facts):
	id2trp = []
	for fact in facts:
		h, r, t = fact
		id2trp.append((h, r, t))

	sp_rel = "SPECIAL_RELATION"

	# create new triplets
	id2trp = remove_duplicate(id2trp)
	trp2id = {trp: idx for idx, trp in enumerate(id2trp)}
	triplets = []
	for fact1 in facts:
		h1, r1, t1 = fact1
		triplet1_ent = f"TRIPLET_{trp2id[fact1]}"
		# check for triplet-triplet relation 
		for fact2 in facts:
			h2, r2, t2 = fact2
			triplet2_ent = f"TRIPLET_{trp2id[fact2]}"
			if t1 == h2:
				triplets.append((triplet1_ent, sp_rel, triplet2_ent))		

		# check for triplet-qualifier relation
		for q, v in facts[fact1]:
			triplets.append((triplet1_ent, q, v))
	
	triplets = remove_duplicate(triplets)
	return id2trp, triplets

def convert_to_triplet_graph(id2trp, triplet):
	ent = []
	facts = dict()
	for h, r, t in triplet:
		ent.append(h)
		if t.startswith("TRIPLET"):
			ent.append(t)
	ent = remove_duplicate(ent)
	for e in ent:
		h, r, t = id2trp[int(e[8:])]
		facts[(h, r, t)] = []
	for h, r, t in triplet:
		if not t.startswith("TRIPLET"):
			facts[id2trp[int(h[8:])]].append((r, t))
	return facts