#!/bin/sh
python3 train_triplets.py --data_path ./data/ --data_name $1 --num_epoch 1000
python3 train_qual.py --data_path ./data/ --data_name $1 --num_epoch 1000