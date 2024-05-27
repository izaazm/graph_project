#!/bin/sh
python3 generate_data.py ./JF17K/full $1 120 160 0.5 0.5
python3 val_test_triplet.py $1
python3 val_test_qual.py $1
python3 check.py $1