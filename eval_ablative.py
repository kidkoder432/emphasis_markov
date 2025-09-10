import pprint
import sys
from winsound import MessageBeep

print(sys.argv)


import generate_fsm as gf
from generate_fsm import *


NUM_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 100

gf.changeRaga(sys.argv[2] if len(sys.argv) > 2 else "amrit")


gen = create_tags()


def run(ground=False):
    avg_acr_1 = 0
    avg_acr_2 = 0
    avg_acr_3 = 0

    acr = [[], [], []]

    total_phrases = 0
    rules_broken = 0
    tags_percent = 0.0

    if ground:
        ns = 1
    else:
        ns = NUM_SAMPLES

    for i in range(ns):
        if ground:
            print("Evaluating ground truth vistaar")

            tags = raga_data["new_tags"]
            phrases_raw = raga_data["new_phrases"]

            phrases = []
            for i, phrase in enumerate(phrases_raw):
                tag = tags[i].split("-")[0]
                time = int(tags[i].split("-")[1])
                phrases.append(f"{tag} {time} {phrase}")
                # print(phrases[-1])
        else:
            print("Generating vistaar", i + 1, end="\r", flush=True)
            phrases, _ = generate_all_phrases(swars_list, convolved_splines, True, gen)

        vistaar_eval, _ = evaluate_all_phrases(phrases, swars_list, convolved_splines)

        phrase_evals = vistaar_eval["phrase_evals"]

        # pprint.pprint(vistaar_eval)

        avg_acr_1 += vistaar_eval["full_acr_lag_1"]
        avg_acr_2 += vistaar_eval["full_acr_lag_2"]
        avg_acr_3 += vistaar_eval["full_acr_lag_3"]

        acr[0].append(vistaar_eval["full_acr_lag_1"])
        acr[1].append(vistaar_eval["full_acr_lag_2"])
        acr[2].append(vistaar_eval["full_acr_lag_3"])

        total_phrases += len(phrase_evals)

        rules_broken += vistaar_eval["num_errors"]
        tags_percent += vistaar_eval["unique_tags_%"]

    avg_acr_1 /= ns
    avg_acr_2 /= ns
    avg_acr_3 /= ns

    print()
    print("Average ACR Lag 1: ", avg_acr_1)
    print("Average ACR Lag 2: ", avg_acr_2)
    print("Average ACR Lag 3: ", avg_acr_3)

    print("Total rules broken: ", rules_broken)
    print("Rules broken per vistaar: ", rules_broken / (ns))
    print("% of unique tags: ", tags_percent / (ns))

    return acr


print("--- Ablative Evaluation I - Phrase Rules ---")


gf.ENABLE_EMPHASIS = True
gf.ENABLE_TAGS = True
gf.ENABLE_HYBRID = True
print("Running with emphasis, tags, and hybrid enabled")
acr_full = run()


gf.ENABLE_EMPHASIS = True
gf.ENABLE_TAGS = True
gf.ENABLE_HYBRID = False
print("Running with emphasis and tags enabled")
acr_tags = run()

gen = []
gf.ENABLE_EMPHASIS = True
gf.ENABLE_TAGS = False
gf.ENABLE_HYBRID = False
print("Running with emphasis enabled")
acr_emphasis = run()

gf.ENABLE_EMPHASIS = False
gf.ENABLE_TAGS = False
gf.ENABLE_HYBRID = False
print("Running with base Markov model")
acr_base = run()

print("Evaluating ground truth vistaar")
run(ground=True)

print("Finished all runs")


print("Running t-tests")
import scipy.stats as stats

# lag 1
stat, p = stats.ttest_ind(acr_full[0], acr_tags[0], equal_var=False)
print("t-test full vs tags (lag 1): ", stat, p)
stat, p = stats.ttest_ind(acr_full[0], acr_emphasis[0], equal_var=False)
print("t-test full vs emphasis (lag 1): ", stat, p)
stat, p = stats.ttest_ind(acr_tags[0], acr_emphasis[0], equal_var=False)
print("t-test tags vs emphasis (lag 1): ", stat, p)
stat, p = stats.ttest_ind(acr_full[0], acr_base[0], equal_var=False)
print("t-test full vs base (lag 1): ", stat, p)

# lag 2
stat, p = stats.ttest_ind(acr_full[1], acr_tags[1], equal_var=False)
print("t-test full vs tags (lag 2): ", stat, p)
stat, p = stats.ttest_ind(acr_full[1], acr_emphasis[1], equal_var=False)
print("t-test full vs emphasis (lag 2): ", stat, p)
stat, p = stats.ttest_ind(acr_tags[1], acr_emphasis[1], equal_var=False)
print("t-test tags vs emphasis (lag 2): ", stat, p)
stat, p = stats.ttest_ind(acr_full[1], acr_base[1], equal_var=False)
print("t-test full vs base (lag 2): ", stat, p)

# lag 3
stat, p = stats.ttest_ind(acr_full[2], acr_tags[2], equal_var=False)
print("t-test full vs tags (lag 3): ", stat, p)
stat, p = stats.ttest_ind(acr_full[2], acr_emphasis[2], equal_var=False)
print("t-test full vs emphasis (lag 3): ", stat, p)
stat, p = stats.ttest_ind(acr_tags[2], acr_emphasis[2], equal_var=False)
print("t-test tags vs emphasis (lag 3): ", stat, p)
stat, p = stats.ttest_ind(acr_full[2], acr_base[2], equal_var=False)
print("t-test full vs base (lag 3): ", stat, p)

MessageBeep()
