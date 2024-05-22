from utils import *
import numpy as np
import random
import igraph
import copy
import time
import os
import gc

class TrainData():
	def __init__(self, path, qual=False, clean_data=True, special_relation=True):
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
				triplet1_ent = f"TRIPLET_{self.trp2id[fact1]}"
				# check for triplet-triplet relation 
				for fact2 in fact_all:
					h2, r2, t2 = fact2
					triplet2_ent = f"TRIPLET_{self.trp2id[fact2]}"
					if t1 == h2:
						if self.special_relation:
							triplets.append((triplet1_ent, sp_rel, triplet2_ent))
						else:
							rel = f"ENTITY_{t1}"
							triplets.append((triplet1_ent, rel, triplet2_ent))
							id2rel.append(rel)
					if t2 == h1:
						if self.special_relation:
							triplets.append((triplet2_ent, sp_rel, triplet1_ent))
						else:
							rel = f"ENTITY_{t2}"
							triplets.append((triplet2_ent, rel, triplet1_ent))
							id2rel.append(rel)				
				id2ent.append(triplet1_ent)
				# check for triplet-qualifier relation
				for q, v in fact_all[fact1]:
					triplets.append((triplet1_ent, q, v))
		else:
			for fact in fact_all:
				h, r, t = fact
				triplets.append((h, r, t))

		# # removing unnecessary triplets
		id2ent = []
		id2rel = []
		for trp in triplets:
			h, r, t = trp
			id2ent.append(h)
			id2ent.append(t)
			id2rel.append(r)

		id2ent = remove_duplicate(id2ent)
		id2rel = remove_duplicate(id2rel)
		random.shuffle(id2ent)
		random.shuffle(id2rel)
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

		# TODO: cek ini diganti apa ngga
		# harus diganti kah ini
		sup = [self.triplets[idx] for idx, tf in enumerate(remaining_triplet_indexes) if tf]
		
		msg = np.array(msg)
		random.shuffle(sup)
		sup = np.array(sup)
		add_num = max(int(self.num_triplets * p) - len(msg), 0)
		if add_num > 0:
			msg = np.concatenate([msg, sup[:add_num]])
		sup = sup[add_num:]

		msg_inv = np.fliplr(msg).copy()
		msg_inv[:,1] += self.num_rel
		msg = np.concatenate([msg, msg_inv])

		return msg, sup

class TestNewData():
	def __init__(self, path, qual=False, data_type="valid", special_relation=True):
		self.path = path
		self.qual = qual
		self.data_type = data_type
		self.special_relation = special_relation
		self.ent2id = None
		self.rel2id = None
		self.id2ent, self.id2rel, self.id2trp, self.msg_triplets, self.sup_triplets, self.filter_dict = self.process_triplet()
		self.num_ent, self.num_rel = len(self.id2ent), len(self.id2rel)

	def read_fact(self, file):			
		triplet = []
		_, _, _, fact_all = read_HKG(self.path + file + ".json")
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

		return id2ent, id2rel, id2trp, fact_all 

	def clean_data(self, triplets):
		id2ent = []
		id2rel = []
		for trp in triplets:
			h, r, t = trp
			id2ent.append(h)
			id2ent.append(t)
			id2rel.append(r)
		return id2ent, id2rel

	def fact_to_triplet(self, trp2id, fact):
		new_triplets, new_entities, new_relations = [], [], []
		if self.special_relation:
			sp_rel = "SPECIAL_RELATION"
			new_relations.append(sp_rel)

		for fact1 in fact:
			h1, r1, t1 = fact1
			triplet1_ent = f"TRIPLET_{trp2id[fact1]}"
			# check for triplet-triplet relation 
			for fact2 in fact:
				h2, r2, t2 = fact2
				triplet2_ent = f"TRIPLET_{trp2id[fact2]}"
				if t1 == h2:
					if self.special_relation:
						new_triplets.append((triplet1_ent, sp_rel, triplet2_ent))
					else:
						rel = f"ENTITY_{t1}"
						new_triplets.append((triplet1_ent, rel, triplet2_ent))
						new_relations.append(rel)
				if t2 == h1:
					if self.special_relation:
						new_triplets.append((triplet2_ent, sp_rel, triplet1_ent))
					else:
						rel = f"ENTITY_{t2}"
						new_triplets.append((triplet2_ent, rel, triplet1_ent))
						new_relations.append(rel)				
			new_entities.append(triplet1_ent)
			# check for triplet-qualifier relation
			for q, v in fact[fact1]:
				new_triplets.append((triplet1_ent, q, v))
			
		return new_entities, new_relations, new_triplets	

	def process_triplet(self):
		total_triplets = []
		id2ent, id2rel, msg_triplets, sup_triplets = [], [], [], []
		_, _, id2trp_msg, fact_msg = self.read_fact("/msg")
		_, _, id2trp_val, fact_val = self.read_fact("/valid")
		_, _, id2trp_test, fact_test = self.read_fact("/test")
		id2trp = remove_duplicate(id2trp_msg + id2trp_val + id2trp_test)

		if self.qual:
			trp2id = {trp: idx for idx, trp in enumerate(id2trp)}
			_, _, new_msg_triplets = self.fact_to_triplet(trp2id, fact_msg)	
			_, _, new_val_triplets = self.fact_to_triplet(trp2id, fact_val)
			_, _, new_test_triplets = self.fact_to_triplet(trp2id, fact_test)

			total_triplets = new_msg_triplets + new_val_triplets + new_test_triplets
			msg_triplets = new_msg_triplets
			id2ent, id2rel = self.clean_data(total_triplets)
			id2ent = remove_duplicate(id2ent)
			id2rel = remove_duplicate(id2rel)
			random.shuffle(id2ent)
			random.shuffle(id2rel)

			self.ent2id = {ent: idx for idx, ent in enumerate(id2ent)}
			self.rel2id = {rel: idx for idx, rel in enumerate(id2rel)}
			num_rel = len(self.rel2id)
			msg_triplets = [(self.ent2id[h], self.rel2id[r], self.ent2id[t]) for h, r, t in msg_triplets]
			msg_inv_triplets = [(t, r+num_rel, h) for h,r,t in msg_triplets]

			if self.data_type == "valid":
				for h, r, t in new_val_triplets:
					sup_triplets.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))
					assert (self.ent2id[h], self.rel2id[r], self.ent2id[t]) not in msg_triplets, (self.ent2id[h], self.rel2id[r], self.ent2id[t])
			elif self.data_type == "test":
				for h, r, t in new_test_triplets:
					sup_triplets.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))
					assert (self.ent2id[h], self.rel2id[r], self.ent2id[t]) not in msg_triplets, (self.ent2id[h], self.rel2id[r], self.ent2id[t])
			else:
				raise ValueError("Data type value is not valid or test")

		else:
			for h, r, t in id2trp_msg:
				msg_triplets.append((h, r, t))
				total_triplets.append((h, r, t))
			for h, r, t in id2trp_val:
				total_triplets.append((h, r, t))
			for h, r, t in id2trp_test:
				total_triplets.append((h, r, t))

			id2ent, id2rel = self.clean_data(total_triplets)
			id2ent = remove_duplicate(id2ent)
			id2rel = remove_duplicate(id2rel)
			random.shuffle(id2ent)
			random.shuffle(id2rel)

			self.ent2id = {ent: idx for idx, ent in enumerate(id2ent)}
			self.rel2id = {rel: idx for idx, rel in enumerate(id2rel)}
			num_rel = len(self.rel2id)
			msg_triplets = [(self.ent2id[h], self.rel2id[r], self.ent2id[t]) for h, r, t in msg_triplets]
			msg_inv_triplets = [(t, r+num_rel, h) for h,r,t in msg_triplets]

			# TODO: cek ini diganti apa ngga
			# harus diganti kah ini
			if self.data_type == "valid":
				for h, r, t in id2trp_val:
					sup_triplets.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))
					assert (self.ent2id[h], self.rel2id[r], self.ent2id[t]) not in msg_triplets, (self.ent2id[h], self.rel2id[r], self.ent2id[t])
			elif self.data_type == "test":
				for h, r, t in id2trp_test:
					sup_triplets.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))
					assert (self.ent2id[h], self.rel2id[r], self.ent2id[t]) not in msg_triplets, (self.ent2id[h], self.rel2id[r], self.ent2id[t])
			else:
				raise ValueError("Data type value is not valid or test")

		filter_dict = {}
		for triplet in total_triplets:
			h, r, t = triplet
			if not self.qual:
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

		return id2ent, id2rel, id2trp, np.array(msg_triplets), np.array(sup_triplets), filter_dict
