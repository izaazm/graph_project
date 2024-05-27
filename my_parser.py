import argparse
import json

def parse(test=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default = "./data/", type = str)
    parser.add_argument('--data_name', default = 'JF17K_50', type = str)
    parser.add_argument('--exp', default = 'exp', type = str)
    parser.add_argument('-m', '--margin', default = 2, type = float)
    parser.add_argument('-lr', '--learning_rate', default=2e-4, type = float)
    parser.add_argument('-nle', '--num_layer_ent', default = 2, type = int)
    parser.add_argument('-nlr', '--num_layer_rel', default = 2, type = int)
    parser.add_argument('-d_e', '--dimension_entity', default = 32, type = int)
    parser.add_argument('-d_r', '--dimension_relation', default = 32, type = int)
    parser.add_argument('-hdr_e', '--hidden_dimension_ratio_entity', default = 8, type = int)
    parser.add_argument('-hdr_r', '--hidden_dimension_ratio_relation', default = 4, type = int)
    parser.add_argument('-b', '--num_bin', default = 10, type = int)
    parser.add_argument('-e', '--num_epoch', default = 10000, type = int)
    parser.add_argument('-sp', '--special_relation', default = True, type = bool)
    parser.add_argument('-v', '--validation_epoch', default = 100, type = int)
    parser.add_argument('--num_head', default = 8, type = int)
    parser.add_argument('--num_neg', default = 10, type = int)

    args = parser.parse_args()

    return args