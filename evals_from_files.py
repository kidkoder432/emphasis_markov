from eval_full import *
import os

import generate_full as gf

directory = './test3/'

gf.changeRaga("jog")
files = [f for f in os.listdir(directory) if f.endswith('.txt') and 'phrases_jog_' in f]
for filename in files:

    with open(directory + filename) as f:
        phrases = f.read().split("\n")
    
    print("Evaluating sample", filename[-7:-4])
    if filename[-7:-4] == "648":
        phrases = get_dataset_vistaar()
        vocab_metric = run_ablative_vocab(phrases, True)
        struct_metric = run_struct_fid(phrases)
    else:
        vocab_metric = run_ablative_vocab([phrases])
        struct_metric = run_struct_fid([phrases])