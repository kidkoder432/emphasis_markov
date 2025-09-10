import pprint
import sys
from collections import Counter, defaultdict
from winsound import MessageBeep

import numpy as np

import generate_fsm as gf

NUM_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 100

gf.changeRaga(sys.argv[2] if len(sys.argv) > 2 else "amrit")


gen = gf.create_tags()

random_pattern = [np.random.choice([-1, 0, 1]) for _ in range(50)]


def evaluate_phrase(phrase_str, swars_list, convolved_splines):
    """Evaluates the quality of a generated phrase."""

    evals = defaultdict(bool)
    tag, emphasis_note = phrase_str.split(" ")[:2]
    phrase_str = phrase_str.split(" ")[3:]
    notes = [s.split("-")[1] for s in phrase_str]

    midi = [gf.swar2midi_map[x] for x in notes]

    diff = np.diff(midi)
    trigrams = []
    for i in range(len(diff) - 2):
        if tuple(diff[i : i + 3]) not in trigrams:
            trigrams.append(tuple(diff[i : i + 3]))

    evals["trigrams"] = len(trigrams)
    return tag, emphasis_note, notes, evals


def evaluate_all_phrases(phrases_list, swars_list, convolved_splines):
    """Evaluates the quality of all generated phrases."""

    evals = {"phrase_evals": []}
    tags = []
    emphasis_notes = []
    notes_lists = []
    for phrase_str in phrases_list:
        tag, emphasis_note, notes, phrase_evals = evaluate_phrase(
            phrase_str, swars_list, convolved_splines
        )
        phrase_evals["tag"] = tag
        phrase_evals["emphasis_note"] = emphasis_note
        evals["phrase_evals"].append(phrase_evals)
        tags.append(tag)
        emphasis_notes.append(emphasis_note)
        notes_lists.append(notes)

    all_notes = [note for notes in notes_lists for note in notes]
    midi = [gf.swar2midi_map[x] for x in all_notes]

    diff = np.diff(midi)
    trigrams = []
    for i in range(len(diff) - 2):
        if tuple(diff[i : i + 3]) not in trigrams:
            trigrams.append(tuple(diff[i : i + 3]))

    evals["vocab_size"] = len(trigrams)
    evals["avg_len"] = len(all_notes) / len(tags)

    if notes_lists[-1][-1] != "S":
        evals["err_last_note"] = True

    num_unique_tags = len(
        set([tags[i] + " " + emphasis_notes[i] for i in range(len(tags))])
    )
    evals["unique_tags_%"] = num_unique_tags / len(gf.unique_note_tags) * 100

    return evals


def get_dataset_phrases():
    tags = gf.raga_data["new_tags"]
    phrases_raw = gf.raga_data["new_phrases"]

    phrases = []
    for i, phrase in enumerate(phrases_raw):
        tag = tags[i].split("-")[0]
        time = int(tags[i].split("-")[1])
        phrase = " ".join([n for n in phrase.split(" ")])
        phrases.append(f"{tag} {time} {phrase}")
        # print(phrases[-1])

    return phrases


def run(ground=False):
    num_errors = 0
    tags_unique = 0
    phrase_len = 0
    vocabs = []

    if ground:
        ns = 1
    else:
        ns = NUM_SAMPLES

    for i in range(ns):
        if ground:
            print("Evaluating ground truth vistaar")
            phrases = get_dataset_phrases()
        else:
            print("Generating vistaar", i + 1, end="\r", flush=True)

            phrases, _ = gf.generate_all_phrases(
                gf.swars_list, gf.convolved_splines, gen=gen
            )

        vistaar_eval = evaluate_all_phrases(
            phrases, gf.swars_list, gf.convolved_splines
        )

        num_errors += vistaar_eval.get("err_last_note", 0)
        tags_unique += vistaar_eval["unique_tags_%"]
        phrase_len += vistaar_eval["avg_len"]
        vocabs.append(vistaar_eval["vocab_size"])

    print("% of unique tags", tags_unique / ns)
    print("Averge phrase` length", phrase_len / ns)
    print("Num errors", num_errors / ns)
    print("Average vocab size", np.mean(vocabs))
    return vocabs


print("--- Ablative Evaluation III - Vocab Size ---")


gf.ENABLE_EMPHASIS = True
gf.ENABLE_TAGS = True
gf.ENABLE_HYBRID = True
print("Running with emphasis, tags, and hybrid enabled")
vocab_full = run()


gf.ENABLE_EMPHASIS = True
gf.ENABLE_TAGS = True
gf.ENABLE_HYBRID = False
print("Running with emphasis and tags enabled")
vocab_tags = run()

gen = []
gf.ENABLE_EMPHASIS = True
gf.ENABLE_TAGS = False
gf.ENABLE_HYBRID = False
print("Running with emphasis enabled")
vocab_emphasis = run()

gf.ENABLE_EMPHASIS = False
gf.ENABLE_TAGS = False
gf.ENABLE_HYBRID = False
print("Running with base Markov model")
vocab_base = run()

print("Evaluating ground truth vistaar")
vocab_ref = run(ground=True)

print("Finished all runs")

reference_vocab = np.array(vocab_ref)

# --- Step 3: Run the t-tests on these new lists of scores ---

from scipy.stats import ttest_ind

print("--- T-Tests of Vocab Size ---")
# Compare all models to each other
p_full_vs_base = ttest_ind(vocab_full, vocab_base, equal_var=False)[1]
print(f"Full Model vs. Base Model p-value: {p_full_vs_base}")

p_full_vs_tags = ttest_ind(vocab_full, vocab_tags, equal_var=False)[1]
print(f"Full Model vs. Tags-Only p-value: {p_full_vs_tags}")

p_full_vs_emphasis = ttest_ind(vocab_full, vocab_emphasis, equal_var=False)[1]
print(f"Full Model vs. Emphasis-Only p-value: {p_full_vs_emphasis}")

p_tags_vs_base = ttest_ind(vocab_tags, vocab_base, equal_var=False)[1]
print(f"Tags-Only vs. Base Model p-value: {p_tags_vs_base}")

p_tags_vs_emphasis = ttest_ind(vocab_tags, vocab_emphasis, equal_var=False)[1]
print(f"Tags-Only vs. Emphasis-Only p-value: {p_tags_vs_emphasis}")

p_emphasis_vs_base = ttest_ind(vocab_emphasis, vocab_base, equal_var=False)[1]
print(f"Emphasis-Only vs. Base Model p-value: {p_emphasis_vs_base}")

MessageBeep()
