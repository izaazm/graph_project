import os
import json
import random
import argparse
from utils import read_HKG

parser = argparse.ArgumentParser()
parser.add_argument('data')
args = parser.parse_args()

full_data = args.data
if full_data not in os.listdir():
    raise ValueError
print(f"\nCHECKING {full_data}")

def intersection(set1, set2):
    return set1 & set2

_, _, _, train_facts = read_HKG(f"{full_data}/train.json")
_, _, _, inference_facts = read_HKG(f"{full_data}/kg_inference.json")

entity_train = set()
entity_inference = set()
relation_train = set()
relation_inference = set()

for fact in train_facts:
    h, r, t = fact
    entity_train.add(h)
    entity_train.add(t)
    relation_train.add(r)
    for qual in train_facts[fact]:
        q, v = qual
        relation_train.add(q)
        entity_train.add(v)

num_fact_existing_entity = 0
num_fact_existing_relation = 0

for fact in inference_facts:
    h, r, t = fact
    entity_inference.add(h)
    entity_inference.add(t)
    relation_inference.add(r)
    for qual in inference_facts[fact]:
        q, v = qual
        relation_inference.add(q)
        entity_inference.add(v)

        if h in entity_train or t in entity_train or v in entity_train:
            num_fact_existing_entity += 1
        if r in relation_train or q in relation_train:
            num_fact_existing_relation += 1

train_fact_with_qual = 0
total_train_qual = 0
for fact in train_facts:
    if len(train_facts[fact]) > 0:
        train_fact_with_qual += 1
        total_train_qual += len(train_facts[fact])

inference_fact_with_qual = 0
total_inference_qual = 0
for fact in inference_facts:
    if len(inference_facts[fact]) > 0:
        inference_fact_with_qual += 1
        total_inference_qual += len(inference_facts[fact])

intersection_entity = intersection(entity_train, entity_inference)
intersection_relation = intersection(relation_train, relation_inference)
percent_fact_new_relation = 1 - num_fact_existing_relation / len(inference_facts)

print(f"Number of facts in train: {len(train_facts)}")
print(f"Number of facts in inference: {len(inference_facts)}")
print(f"Number of entities in train: {len(entity_train)}")
print(f"Number of entities in inference: {len(entity_inference)}")
print(f"Number of relations in train: {len(relation_train)}")
print(f"Number of relations in inference: {len(relation_inference)}")
print("\n")
print(f"Number of facts with qual in train: {train_fact_with_qual}")
print(f"Total number of qual in train: {total_train_qual}")
print(f"Number of facts with qual in inference: {inference_fact_with_qual}")
print(f"Total number of qual in inference: {total_inference_qual}")
print("\n")
print(f"Number of facts with new relation: {len(inference_facts) - num_fact_existing_relation}")
print(f"Percentage of facts with new relation: {percent_fact_new_relation}")
