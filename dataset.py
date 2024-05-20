from utils import *
import numpy as np
import random
import igraph
import copy
import time
import os
import gc

class TrainData():
	def __init__(self, path, qual=False, clean_data=False, special_relation=True):
		self.path = path
		self.qual = qual
		self.clean_data = clean_data
		self.special_relation = special_relation
		self.rel_info = {}
		self.pair_info = {}
		self.spanning = []
		self.remaining = []
		self.ent2id = None
		self.rel2id = None
		self.trp2id = None
		self.id2ent, self.id2rel, self.id2trp, self.triplets = self.read_fact(path + '/train.json')
		self.num_triplets = len(self.triplets)
		self.num_ent, self.num_rel = len(self.id2ent), len(self.id2rel)
		
	def read_fact(self, path):
		_, _, _, fact_all = read_HKG(path)
		id2ent, id2rel, id2trp = [], [], []
		for fact in fact_all:
			h, r, t = fact
			id2ent.append(h)
			id2ent.append(t)
			id2rel.append(r)
			id2trp.append((h, r, t))
			if self.qual:
				for (q, v) in fact_all[(h, r, t)]:
					id2ent.append(v)
					id2rel.append(q)

		if self.qual and self.special_relation:
			sp_rel = "SPECIAL_RELATION"
			id2rel.append(sp_rel)

		# create new triplets
		id2trp = remove_duplicate(id2trp)
		self.trp2id = {trp: idx for idx, trp in enumerate(id2trp)}
		triplets = []
		if self.qual:
			for fact1 in fact_all:
				h1, r1, t1 = fact1
				triplet1_ent = f"triplet_{self.trp2id[fact1]}"
				# check for triplet-triplet relation 
				for fact2 in fact_all:
					h2, r2, t2 = fact2
					triplet2_ent = f"triplet_{self.trp2id[fact2]}"
					if t1 == h2:
						if self.special_relation:
							triplets.append(triplet1_ent, sp_rel, triplet2_ent)
						else:
							rel = f"ENTITY_{t1}"
							triplets.append(triplet1_ent, sp_rel, triplet2_ent)
							id2rel.append(rel)
					if t2 == h1:
						if self.special_relation:
							triplets.append(triplet2_ent, sp_rel, triplet1_ent)
						else:
							rel = f"ENTITY_{t1}"
							triplets.append(triplet2_ent, rel, triplet1_ent)
							id2rel.append(rel)				
				id2ent.append(triplet1_ent)
				# check for triplet-qualifier relation
				for q, v in fact_all[fact1]:
					triplets.append(triplet1_ent, q, v)
		else:
			for fact in fact_all:
				h, r, t = fact
				triplets.append((h, r, t))

		# cremoving unnecessary triplets
		if self.clean_data:
			id2ent = []
			id2rel = []
			for trp in triplets:
				h, r, t = trp
				id2ent.append(h)
				id2ent.append(t)
				id2rel.append(r)

		id2ent = remove_duplicate(id2ent)
		id2rel = remove_duplicate(id2rel)
		triplets = remove_duplicate(triplets)
		self.ent2id = {ent: idx for idx, ent in enumerate(id2ent)}
		self.rel2id = {rel: idx for idx, rel in enumerate(id2rel)}
		triplets = [(self.ent2id[h], self.rel2id[r], self.ent2id[t]) for h, r, t in triplets]
		for (h,r,t) in triplets:
			if (h,t) in self.rel_info:
				self.rel_info[(h,t)].append(r)
			else:
				self.rel_info[(h,t)] = [r]
			if r in self.pair_info:
				self.pair_info[r].append((h,t))
			else:
				self.pair_info[r] = [(h,t)]
		G = igraph.Graph.TupleList(np.array(triplets)[:, 0::2])
		G_ent = igraph.Graph.TupleList(np.array(triplets)[:, 0::2], directed = True)
		spanning = G_ent.spanning_tree()
		G_ent.delete_edges(spanning.get_edgelist())
		
		for e in spanning.es:
			e1,e2 = e.tuple
			e1 = spanning.vs[e1]["name"]
			e2 = spanning.vs[e2]["name"]
			self.spanning.append((e1,e2))
		
		spanning_set = set(self.spanning)
		
		print("-----Train Data Statistics-----")
		print(f"{len(self.ent2id)} entities, {len(self.rel2id)} relations")
		print(f"{len(triplets)} triplets")
		self.triplet2idx = {triplet:idx for idx, triplet in enumerate(triplets)}
		self.triplets_with_inv = np.array([(t, r + len(id2rel), h) for h,r,t in triplets] + triplets)
		return id2ent, id2rel, id2trp, triplets

	def split_transductive(self, p):
		msg, sup = [], []

		rels_encountered = np.zeros(self.num_rel)
		remaining_triplet_indexes = np.ones(self.num_triplets)

		for h,t in self.spanning:
			r = random.choice(self.rel_info[(h,t)])
			msg.append((h, r, t))
			remaining_triplet_indexes[self.triplet2idx[(h,r,t)]] = 0
			rels_encountered[r] = 1


		for r in (1-rels_encountered).nonzero()[0].tolist():
			h,t = random.choice(self.pair_info[int(r)])
			msg.append((h, r, t))
			remaining_triplet_indexes[self.triplet2idx[(h,r,t)]] = 0

		start = time.time()
		sup = [self.triplets[idx] for idx, tf in enumerate(remaining_triplet_indexes) if tf]
		
		msg = np.array(msg)
		random.shuffle(sup)
		sup = np.array(sup)
		add_num = max(int(self.num_triplets * p) - len(msg), 0)
		msg = np.concatenate([msg, sup[:add_num]])
		sup = sup[add_num:]

		msg_inv = np.fliplr(msg).copy()
		msg_inv[:,1] += self.num_rel
		msg = np.concatenate([msg, msg_inv])

		return msg, sup

class TestNewData():
	def __init__(self, path, qual=False, data_type="valid"):
		self.path = path
		self.qual = qual
		self.data_type = data_type
		self.ent2id = None
		self.rel2id = None
		self.trp2id = None
		self.id2ent, self.id2rel, self.id2trp, self.msg_triplets, self.sup_triplets, self.filter_dict = self.read_triplet()
		self.num_ent, self.num_rel = len(self.id2ent), len(self.id2rel)
		

	def read_triplet(self):
		id2ent, id2rel, msg_triplets, sup_triplets = [], [], [], []
		total_triplets = []
		_, _, _, fact_all = read_HKG(self.path + 'msg.json')
		id2ent, id2rel, id2trp = [], [], []
		for fact in fact_all:
			h, r, t = fact
			id2ent.append(h)
			id2ent.append(t)
			id2rel.append(r)
			id2trp.append((h, r, t))
			if self.qual:
				for (q, v) in fact_all[(h, r, t)]:
					id2ent.append(v)
					id2rel.append(q)

		with open(self.path + "msg.txt", 'r') as f:
			for line in f.readlines():
				h, r, t = line.strip().split('\t')
				id2ent.append(h)
				id2ent.append(t)
				id2rel.append(r)
				msg_triplets.append((h, r, t))
				total_triplets.append((h, r, t))

		id2ent = remove_duplicate(id2ent)
		id2rel = remove_duplicate(id2rel)
		self.ent2id = {ent: idx for idx, ent in enumerate(id2ent)}
		self.rel2id = {rel: idx for idx, rel in enumerate(id2rel)}
		num_rel = len(self.rel2id)
		msg_triplets = [(self.ent2id[h], self.rel2id[r], self.ent2id[t]) for h, r, t in msg_triplets]
		msg_inv_triplets = [(t, r+num_rel, h) for h,r,t in msg_triplets]

		with open(self.path + self.data_type + ".txt", 'r') as f:
			for line in f.readlines():
				h, r, t = line.strip().split('\t')
				sup_triplets.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))
				assert (self.ent2id[h], self.rel2id[r], self.ent2id[t]) not in msg_triplets, \
					(self.ent2id[h], self.rel2id[r], self.ent2id[t]) 
				total_triplets.append((h,r,t))		
		for data_type in ['valid', 'test']:
			if data_type == self.data_type:
				continue
			with open(self.path + data_type + ".txt", 'r') as f:
				for line in f.readlines():
					h, r, t = line.strip().split('\t')
					assert (self.ent2id[h], self.rel2id[r], self.ent2id[t]) not in msg_triplets, \
						(self.ent2id[h], self.rel2id[r], self.ent2id[t]) 
					total_triplets.append((h,r,t))	


		filter_dict = {}
		for triplet in total_triplets:
			h,r,t = triplet
			if ('_', self.rel2id[r], self.ent2id[t]) not in filter_dict:
				filter_dict[('_', self.rel2id[r], self.ent2id[t])] = [self.ent2id[h]]
			else:
				filter_dict[('_', self.rel2id[r], self.ent2id[t])].append(self.ent2id[h])

			if (self.ent2id[h], '_', self.ent2id[t]) not in filter_dict:
				filter_dict[(self.ent2id[h], '_', self.ent2id[t])] = [self.rel2id[r]]
			else:
				filter_dict[(self.ent2id[h], '_', self.ent2id[t])].append(self.rel2id[r])
				
			if (self.ent2id[h], self.rel2id[r], '_') not in filter_dict:
				filter_dict[(self.ent2id[h], self.rel2id[r], '_')] = [self.ent2id[t]]
			else:
				filter_dict[(self.ent2id[h], self.rel2id[r], '_')].append(self.ent2id[t])
		
		print(f"-----{self.data_type.capitalize()} Data Statistics-----")
		print(f"Message set has {len(msg_triplets)} triplets")
		print(f"Supervision set has {len(sup_triplets)} triplets")
		print(f"{len(self.ent2id)} entities, " + \
			  f"{len(self.rel2id)} relations, "+ \
			  f"{len(total_triplets)} triplets")

		msg_triplets = msg_triplets + msg_inv_triplets

		return id2ent, id2rel, np.array(msg_triplets), np.array(sup_triplets), filter_dict
