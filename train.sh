#!/bin/sh
python3 train_triplets.py --data_path ./data/ --data_name $1 --num_epoch 8000 --save_method $2
python3 train_qual.py --data_path ./data/ --data_name $1 --num_epoch 8000 --save_method $2