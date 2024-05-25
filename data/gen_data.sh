#!/bin/sh
python3 generate_data.py ./JF17K/full JF17K_50 120 150 0.5 0.4
python3 val_test.py JF17K_50
python3 check.py JF17K_50