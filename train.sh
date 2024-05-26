#!/bin/sh
python3 train_triplets.py --data_path ./data/ --data_name $1 --num_epoch 8000 --special_relation $2
python3 train_qual.py --data_path ./data/ --data_name $1 --num_epoch 8000 --special_relation $2