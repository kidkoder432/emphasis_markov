import pprint, re
from winsound import MessageBeep

import generate_fsm as gf
from generate_fsm import *

NUM_SAMPLES = 50

def run():
    avg_acr_1 = 0
    avg_acr_2 = 0
    avg_acr_3 = 0

    total_phrases = 0
    rules_broken = 0
    tags_percent = 0.0

    for i in range(NUM_SAMPLES):
        print("Generating vistaar", i + 1, end="\r", flush=True)

        phrases, _ = generate_all_phrases(swars_list, convolved_splines, True)

        vistaar_eval, _ = evaluate_all_phrases(phrases, swars_list, convolved_splines)

        phrase_evals = vistaar_eval["phrase_evals"]

        avg_acr_1 += vistaar_eval["full_acr_lag_1"]
        avg_acr_2 += vistaar_eval["full_acr_lag_2"]
        avg_acr_3 += vistaar_eval["full_acr_lag_3"]

        total_phrases += len(phrase_evals)

        rules_broken += vistaar_eval["num_errors"]
        tags_percent += vistaar_eval["unique_tags_%"]
    print()
    avg_acr_1 /= NUM_SAMPLES
    avg_acr_2 /= NUM_SAMPLES
    avg_acr_3 /= NUM_SAMPLES
    print("Average ACR Lag 1: ", avg_acr_1)
    print("Average ACR Lag 2: ", avg_acr_2)
    print("Average ACR Lag 3: ", avg_acr_3)

    print("Total rules broken: ", rules_broken)
    print("Rules broken per vistaar: ", rules_broken / NUM_SAMPLES)
    print("% of unique tags: ", tags_percent / NUM_SAMPLES)

def run_ground():
    avg_acr_1 = 0
    avg_acr_2 = 0
    avg_acr_3 = 0

    total_phrases = 0
    rules_broken = 0
    tags_percent = 0.0

    print("Evaluating ground truth vistaar")

    tags = amrit_data["new_tags"]
    phrases_raw = amrit_data["new_phrases"]

    phrases = []
    for i, phrase in enumerate(phrases_raw):
        tag = tags[i].split("-")[0]
        time = int(tags[i].split("-")[1])
        phrase = ' '.join([re.sub(r"(\d+)(.+)", r"\1-\2", n) for n in phrase.split(" ")])
        phrases.append(f"{tag} {time} {phrase}")
        # print(phrases[-1])


    vistaar_eval, _ = evaluate_all_phrases(phrases, swars_list, convolved_splines)

    phrase_evals = vistaar_eval["phrase_evals"]

    # pprint.pprint(vistaar_eval)

    avg_acr_1 += vistaar_eval["full_acr_lag_1"]
    avg_acr_2 += vistaar_eval["full_acr_lag_2"]
    avg_acr_3 += vistaar_eval["full_acr_lag_3"]

    total_phrases += len(phrase_evals)

    rules_broken += vistaar_eval["num_errors"]
    tags_percent += vistaar_eval["unique_tags_%"]
    print("Average ACR Lag 1: ", avg_acr_1)
    print("Average ACR Lag 2: ", avg_acr_2)
    print("Average ACR Lag 3: ", avg_acr_3)

    print("Total rules broken: ", rules_broken)
    print("Rules broken per vistaar: ", rules_broken)
    print("% of unique tags: ", tags_percent)

gf.ENABLE_EMPHASIS = True
gf.ENABLE_TAGS = True
gf.ENABLE_HYBRID = True
print("Running with emphasis, tags, and hybrid enabled")
run()


gf.ENABLE_EMPHASIS = True
gf.ENABLE_TAGS = True
gf.ENABLE_HYBRID = False
print("Running with emphasis and tags enabled")
run()


gf.ENABLE_EMPHASIS = True
gf.ENABLE_TAGS = False
gf.ENABLE_HYBRID = False
print("Running with emphasis enabled")
run()

print("Evaluating ground truth vistaar")
run_ground()

print("Finished all runs")
MessageBeep()
