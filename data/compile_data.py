import os

out_file_path = "./JF17K_25/msg.json"
data_dir = './JF17K'

if os.path.exists(out_file_path):
    os.remove(out_file_path)
else:
    print(f"File '{out_file_path}' does not exist.")

total_fact_write = 0
with open(out_file_path, "w") as out_file:
    file_list = os.listdir(data_dir)
    for file in file_list:
        with open(os.path.join(data_dir, file), 'r') as json_file:
            for fact in json_file:
                out_file.write(fact)
                total_fact_write += 1
    out_file.close()

# check
total_fact_read = 0
with open(out_file_path, "r") as out_file:
    for fact in out_file:
        total_fact_read += 1


assert total_fact_write == total_fact_read
print(f"Compile finished, Total facts = {total_fact_read}")