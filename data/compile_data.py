import os
import json

out_path = "./JF17K_25"
data_dir = './JF17K'

if os.path.exists(os.path.join(out_path, "full.json")):
    os.remove(os.path.join(out_path, "full.json"))

n_ent = 0
n_rel = 0
entities = {}
relations = {} 
total_fact_write = 0
fact_seen = set()

def add_ent(x):
    global n_ent
    if not x in entities:
        entities[x] = n_ent
        n_ent += 1

def add_rel(x):
    global n_rel
    if not x in relations:
        relations[x] = n_rel
        n_rel += 1

with open(os.path.join(out_path, "full.json"), "w") as out_file:
    file_list = os.listdir(data_dir)
    for file in file_list:
        with open(os.path.join(data_dir, file), 'r') as json_file:
            for fact in json_file:
                if fact not in fact_seen: # not a duplicate
                    out_file.write(fact)
                    fact_seen.add(fact)
                    total_fact_write += 1

                    cur_fact = json.loads(fact)
                    for rel in cur_fact:
                        if rel == 'N':
                            continue
                        
                        add_rel(rel)
                        if isinstance(cur_fact[rel], list):
                            for ent in cur_fact[rel]:
                                add_ent(ent)
                        else:
                            add_ent(cur_fact[rel])
                    
    out_file.close()

# check
total_fact_read = 0
with open(os.path.join(out_path, "full.json"), "r") as out_file:
    for fact in out_file:
        total_fact_read += 1

assert total_fact_write == total_fact_read
print(f"Compile finished, Total facts = {total_fact_read}")

# Create relation and entities dictionary

if os.path.exists(os.path.join(out_path, "rel2id.txt")):
    os.remove(os.path.join(out_path, "rel2id.txt"))

with open(os.path.join(out_path, "rel2id.txt"), 'w') as file:
    for key, value in relations.items():
        file.write(f"{key} {value}\n")

if os.path.exists(os.path.join(out_path, "ent2id.txt")):
    os.remove(os.path.join(out_path, "ent2id.txt"))

with open(os.path.join(out_path, "ent2id.txt"), 'w') as file:
    for key, value in entities.items():
        file.write(f"{key} {value}\n")

print(f"Id finished, Total relations = {len(relations)}, Total Entities = {len(entities)}")