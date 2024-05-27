import os
import json
import random
import argparse
import igraph
from utils import read_HKG

parser = argparse.ArgumentParser()
parser.add_argument('data')
parser.add_argument('--seed', default = 5, type = int)
args = parser.parse_args()
random.seed(args.seed)

full_data = args.data
if full_data not in os.listdir():
    raise ValueError
print(f"PROCESSING {full_data} TRIPLET")
test = []
test_graph = []
test_rel = set()
test_r2ht = {}
test_q = {}
test_hcon = {}
test_tcon = {}

_, _, _, fact_all = read_HKG(f"{full_data}/kg_inference.json")

for fact in fact_all:
    h, r, t = fact
    test.append((h, r, t))
    if r in test_r2ht:
        test_r2ht[r].append((h, t))
    else:
        test_r2ht[r] = [(h, t)]
    if (h, '_', t) in test_q:
        test_q[(h, '_', t)].append(r)
    else:
        test_q[(h, '_', t)] = [r]
    test_rel.add(r)
    test_graph.append((h, t))
G_test = igraph.Graph.TupleList(test_graph, directed = True)
spanning_test = G_test.spanning_tree()

num_test = len(test)
test_msg = set()
test = set(test)

for e in spanning_test.es:
    h,t = e.tuple
    h = spanning_test.vs[h]["name"]
    t = spanning_test.vs[t]["name"]
    r = random.choice(test_q[(h, '_', t)])
    test_msg.add((h, r, t))
    test_rel.discard(r)
    test.discard((h, r, t))
for r in test_rel:
    h,t = random.choice(test_r2ht[r])
    test_msg.add((h, r, t))
    test.discard((h, r, t))
left_test = sorted(list(test))
test_msg = sorted(list(test_msg))
random.shuffle(left_test)
remainings = int(num_test * 0.6) - len(test_msg)
test_msg += left_test[:remainings]
left_test = left_test[remainings:]

final_valid = left_test[:len(left_test) // 2]
final_test = left_test[len(left_test) // 2:]

print("Number of facts in message KG:", len(test_msg))
print("Number of facts in validation KG:", len(final_valid))
print("Number of facts in test KG:", len(final_test))

if not os.path.exists(f"./{full_data}/triplet/"):
    os.makedirs(f"./{full_data}/triplet/")
    
with open(f"{full_data}/triplet/msg.json", "w") as f:
    for h, r, t in test_msg:
        n = 2
        fact = dict()
        fact[r] = [h, t]
        for q, v in fact_all[(h, r, t)]:
            fact[q] = v
            n += 1
        fact["N"] = n
        f.write(json.dumps(fact) + '\n')

with open(f"{full_data}/triplet/valid.json", "w") as f:
    for h, r, t in final_valid:
        n = 2
        fact = dict()
        fact[r] = [h, t]
        for q, v in fact_all[(h, r, t)]:
            fact[q] = v
            n += 1
        fact["N"] = n
        f.write(json.dumps(fact) + '\n')

with open(f"{full_data}/triplet/test.json", "w") as f:
    for h, r, t in final_test:
        n = 2
        fact = dict()
        fact[r] = [h, t]
        for q, v in fact_all[(h, r, t)]:
            fact[q] = v
            n += 1
        fact["N"] = n
        f.write(json.dumps(fact) + '\n')