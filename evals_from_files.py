from eval_full import *

import generate_full as gf
gf.changeRaga("jog")

for sampleNum in [393, 423, 450, 597, 648]:
    filename = f"./human_eval/text/phrases_jog_{sampleNum}.txt"

    with open(filename) as f:
        phrases = f.read().split("\n")
    
    print("Evaluating sample", sampleNum)
    if sampleNum == 648:
        phrases = get_dataset_vistaar()
        vocab_metric = run_ablative_vocab(phrases, True)
        struct_metric = run_struct_fid(phrases)
    else:
        vocab_metric = run_ablative_vocab([phrases])
        struct_metric = run_struct_fid([phrases])