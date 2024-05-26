#!/bin/sh
python3 generate_data.py ./JF17K/full JF17K_25 100 160 0.3 0.25
python3 val_test.py JF17K_25
python3 check.py JF17K_25