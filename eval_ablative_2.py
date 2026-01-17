import pprint
import sys
from collections import Counter, defaultdict
from winsound import MessageBeep

import numpy as np

import generate_full as gf

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

    diffs = np.diff(midi).clip(-1, 1)
    l = len(diffs)

    patterns = [
        [1] * l,  # up
        [-1] * l,  # down
        [1] * (l // 2) + [-1] * (l - l // 2),  # mountain
        [-1] * (l // 2) + [1] * (l - l // 2),  # valley
        [0] * l,  # flat
        ([1, -1] * (l // 2 + 1))[:l],  # alternating I
        ([-1, 1] * (l // 2 + 1))[:l],  # alternating II
    ]

    similarity = [np.linalg.norm(diffs - p) / l for p in patterns]
    evals["similarity"] = similarity
    evals["best_match"] = np.argmin(similarity)

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

    dist = Counter([p["best_match"] for p in evals["phrase_evals"]])
    for i in range(7):
        if i not in dist:
            dist[i] = 0
    evals["dist"] = [d / dist.total() for d in dist.values()]

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
    dists = []

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
        dists.append([d / sum(vistaar_eval["dist"]) for d in vistaar_eval["dist"]])

    print(len(dists), [len(d) for d in dists])
    print("% of unique tags", tags_unique / NUM_SAMPLES)
    print("Num errors", num_errors / NUM_SAMPLES)
    return dists


print("--- Ablative Evaluation II - Pattern Distribution ---")


gf.ENABLE_EMPHASIS = True
gf.ENABLE_TAGS = True
gf.ENABLE_HYBRID = True
print("Running with emphasis, tags, and hybrid enabled")
dist_full = run()


gf.ENABLE_EMPHASIS = True
gf.ENABLE_TAGS = True
gf.ENABLE_HYBRID = False
print("Running with emphasis and tags enabled")
dist_tags = run()

gen = []
gf.ENABLE_EMPHASIS = True
gf.ENABLE_TAGS = False
gf.ENABLE_HYBRID = False
print("Running with emphasis enabled")
dist_emphasis = run()

gf.ENABLE_EMPHASIS = False
gf.ENABLE_TAGS = False
gf.ENABLE_HYBRID = False
print("Running with base Markov model")
dist_base = run()

print("Evaluating ground truth vistaar")
dist_ref = run(ground=True)

print("Finished all runs")

reference_dist = np.array(dist_ref[0])

# --- Step 2: Calculate the KL Divergence scores for each model vs. the reference ---


def kl_divergence(p, q):
    p += 1e-10
    q += 1e-10
    return np.sum(p * np.log(p / q))


def calculate_kl_scores(generated_dists, ref_dist):
    """Calculates a list of KL divergence scores for each run against the reference."""
    scores = []
    for gen_dist in generated_dists:
        # Calculate KL divergence of this single run from the reference
        score = kl_divergence(np.array(gen_dist), ref_dist)
        scores.append(score)
    return scores


# Calculate the scores for each model
kl_scores_full = calculate_kl_scores(dist_full, reference_dist)
kl_scores_tags = calculate_kl_scores(dist_tags, reference_dist)
kl_scores_emphasis = calculate_kl_scores(dist_emphasis, reference_dist)
kl_scores_base = calculate_kl_scores(dist_base, reference_dist)


# --- Step 3: Run the t-tests on these new lists of scores ---

from scipy.stats import ttest_ind

print("--- T-Tests of KL Divergence from Ground Truth ---")

# Compare your structured models to the baseline
p_full_vs_base = ttest_ind(kl_scores_full, kl_scores_base, equal_var=False)[1]
print(f"Full Model vs. Base Model p-value: {p_full_vs_base}")

# Compare your structured models to each other
p_full_vs_tags = ttest_ind(kl_scores_full, kl_scores_tags, equal_var=False)[1]
print(f"Full Model vs. Tags-Only p-value: {p_full_vs_tags}")

p_full_vs_emphasis = ttest_ind(kl_scores_full, kl_scores_emphasis, equal_var=False)[1]
print(f"Full Model vs. Emphasis-Only p-value: {p_full_vs_emphasis}")
MessageBeep()
