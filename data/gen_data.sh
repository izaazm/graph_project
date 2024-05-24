#!/bin/sh
python3 generate_data.py ./JF17K_25/full JF17K_25 120 160 0.4 0.5
python3 val_test.py JF17K_25