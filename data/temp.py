from utils import *

entity_all, relation_all, triplet_all, fact_all = read_HKG("./JF17K_25/full.json")

print(len(entity_all), len(relation_all), len(triplet_all), len(fact_all))

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
	# else:
	# 	print(h, r, t)
	# 	for (q, v) in fact_all[(h, r, t)]:
	# 		print(q, v)
	# 	print()

entity = remove_duplicate(entity)
relation = remove_duplicate(relation)
fact = remove_duplicate_facts(fact)

print(len(entity), len(relation), len(fact))

num_relation = len(relation)
random.shuffle(relation)
relation_test = set(relation[:int(num_relation * 0.25)])
relation_train = set(relation[int(num_relation * 0.25):])

seed_train = random.sample(entity, args.n_train)
entity_train = sample_2hop(triplet, seed_train, 50)