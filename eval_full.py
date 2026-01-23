from pprint import pprint

import sys
from collections import Counter, defaultdict
from winsound import MessageBeep

from scipy.stats import ttest_ind

import numpy as np

import generate_full as gf

from training.learn_emphasis import *

NUM_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 100

gf.changeRaga(sys.argv[2] if len(sys.argv) > 2 else "amrit")


gen = gf.create_tags()

random_pattern = [np.random.choice([-1, 0, 1]) for _ in range(50)]


def calc_vocab(phrase_str, swars_list, convolved_splines):
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


def calc_vocab_all(phrases_list, swars_list, convolved_splines):
    """Evaluates the quality of all generated phrases."""

    evals = {"phrase_evals": []}
    tags = []
    emphasis_notes = []
    notes_lists = []
    for phrase_str in phrases_list:
        tag, emphasis_note, notes, phrase_evals = calc_vocab(
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


def get_dataset_vistaar():
    print("Retrieving dataset vistaar...", file=sys.stderr)

    tags = gf.raga_data["new_tags"]
    phrases_raw = gf.raga_data["new_phrases"]

    phrases = []
    for i, phrase in enumerate(phrases_raw):
        tag = tags[i].split("-")[0]
        time = int(tags[i].split("-")[1])
        phrase = " ".join([n for n in phrase.split(" ")])
        phrases.append(f"{tag} {time} {phrase}")
        # print(phrases[-1])

    return [phrases]

def generate_vistaars(num_samples=NUM_SAMPLES):
    vistaars = []
    for i in range(num_samples): 
        print("Generating vistaar", i + 1, end="\r", flush=True, file=sys.stderr)
        phrases, _ = gf.generate_all_phrases(gf.swars_list, gf.convolved_splines, gen=gen)
        vistaars.append(phrases)
    return vistaars

def run_ablative_vocab(vistaars, ground=False):
    num_errors = 0
    tags_unique = 0
    phrase_len = []
    vocabs = []

    ns = len(vistaars)

    for i in range(ns):
        phrases = vistaars[i]

        vistaar_eval = calc_vocab_all(
            phrases, gf.swars_list, gf.convolved_splines
        )

        num_errors += vistaar_eval.get("err_last_note", 0)
        tags_unique += vistaar_eval["unique_tags_%"]
        phrase_len.append(vistaar_eval["avg_len"])
        vocabs.append(vistaar_eval["vocab_size"])

    print("% of unique tags", tags_unique / ns)
    print("Average phrase length", np.mean(phrase_len))
    print("Num errors", num_errors / ns)
    print("Average vocab size", np.mean(vocabs))
    print("Std phrase length", np.std(phrase_len))
    print("Std vocab size", np.std(vocabs))
    return vocabs


notes = gf.raga_data["notes"]

t_ref = gf.splines_data["t"]
splines_ref = gf.splines_data["convolved_splines"]


def calculate_features(t, splines):
    out = splines(t)
    return np.array(
        [
            np.mean(out),
            np.std(out),
            np.argmax(out),
            1 - np.count_nonzero(out) / len(out),
        ]
    )


def dist(ta, a, tb, b):
    return np.linalg.norm(calculate_features(ta, a) - calculate_features(tb, b))


def run_struct_fid(vistaars):
    metric = np.zeros((len(notes), len(vistaars)))
    for j in range(len(vistaars)):
        phrases = vistaars[j]
        for i, p in enumerate(phrases):
            phrases[i] = " ".join(p.split(" ")[3:])
        phrase_data = {
            "notes": notes,
            "phrases": phrases,
        }

        t_gen, _, splines_gen = generate_splines(calculate_emphasis_df(phrase_data))
        for i, _ in enumerate(notes):
            metric[i, j] = dist(t_ref, splines_ref[i], t_gen, splines_gen[i])

    pprint(np.mean(metric, axis=1))
    pprint(np.std(metric, axis=1))
    return metric

if __name__ == "__main__":

    dataset = get_dataset_vistaar()
    print([calculate_features(t_ref, splines_ref[i]) for i in range(len(notes))])

    print("Running with emphasis, tags, and hybrid enabled")
    gf.ENABLE_EMPHASIS = True
    gf.ENABLE_TAGS = True
    gf.ENABLE_HYBRID = True
    vistaars = generate_vistaars()

    vocab_full = run_ablative_vocab(vistaars)
    struct_full = run_struct_fid(vistaars)

    print("Running with emphasis and tags enabled")
    gf.ENABLE_EMPHASIS = True
    gf.ENABLE_TAGS = True
    gf.ENABLE_HYBRID = False
    vistaars = generate_vistaars()

    vocab_tags = run_ablative_vocab(vistaars)
    struct_tags = run_struct_fid(vistaars)

    print("Running with emphasis enabled")
    gen = []
    gf.ENABLE_EMPHASIS = True
    gf.ENABLE_TAGS = False
    gf.ENABLE_HYBRID = False
    vistaars = generate_vistaars()

    vocab_emphasis = run_ablative_vocab(vistaars)
    struct_emphasis = run_struct_fid(vistaars)

    print("Running with base Markov model")
    gf.ENABLE_EMPHASIS = False
    gf.ENABLE_TAGS = False
    gf.ENABLE_HYBRID = False
    vistaars = generate_vistaars()

    vocab_base = run_ablative_vocab(vistaars)
    struct_base = run_struct_fid(vistaars)

    vocab_ground = run_ablative_vocab(dataset, ground=True)

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


    print("--- T-Tests of Structural Fidelity ---")

    for i in range(len(notes)):
        t_test_vector = [struct_full[i], struct_tags[i], struct_emphasis[i], struct_base[i]]
        print("length of each t test vector:", len(t_test_vector[0]))

        p_full_tags = ttest_ind(t_test_vector[0], t_test_vector[1], equal_var=False)[1]
        p_full_emphasis = ttest_ind(t_test_vector[0], t_test_vector[2], equal_var=False)[1]
        p_full_base = ttest_ind(t_test_vector[0], t_test_vector[3], equal_var=False)[1]
        p_tags_emphasis = ttest_ind(t_test_vector[1], t_test_vector[2], equal_var=False)[1]
        p_tags_base = ttest_ind(t_test_vector[1], t_test_vector[3], equal_var=False)[1]
        p_emphasis_base = ttest_ind(t_test_vector[2], t_test_vector[3], equal_var=False)[1]

        print("note", i, "full vs base:", p_full_base)
        print("note", i, "full vs tags:", p_full_tags)
        print("note", i, "full vs emphasis:", p_full_emphasis)
        print("note", i, "tags vs emphasis:", p_tags_emphasis)
        print("note", i, "tags vs base:", p_tags_base)
        print("note", i, "emphasis vs base:", p_emphasis_base)


    MessageBeep()
