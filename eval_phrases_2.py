from pprint import pprint
from winsound import MessageBeep

import scipy as sp
from scipy.stats import entropy, ttest_ind

import generate_full as gf
from generate_full import *
from training.learn_emphasis import *

NUM_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 100

gf.changeRaga(sys.argv[2] if len(sys.argv) > 2 else "amrit")


gen = create_tags()

notes = raga_data["notes"]

t_ref = splines_data["t"]
splines_ref = splines_data["convolved_splines"]


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


print([calculate_features(t_ref, splines_ref[i]) for i in range(len(notes))])


def dist(ta, a, tb, b):
    return np.linalg.norm(calculate_features(ta, a) - calculate_features(tb, b))


def run():
    metric = np.zeros((len(notes), NUM_SAMPLES))
    for j in range(NUM_SAMPLES):
        print("Generating vistaar", j + 1, end="\r", flush=True)
        phrases, _ = generate_all_phrases(swars_list, convolved_splines, True, gen)
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


print("--- Structural Fidelity Evaluation ---")

gf.ENABLE_EMPHASIS = True
gf.ENABLE_TAGS = True
gf.ENABLE_HYBRID = True
print("Running with emphasis, tags, and hybrid enabled")
metric_full = run()


gf.ENABLE_EMPHASIS = True
gf.ENABLE_TAGS = True
gf.ENABLE_HYBRID = False
print("Running with emphasis and tags enabled")
metric_tags = run()

gen = []
gf.ENABLE_EMPHASIS = True
gf.ENABLE_TAGS = False
gf.ENABLE_HYBRID = False
print("Running with emphasis enabled")
metric_emphasis = run()

gf.ENABLE_EMPHASIS = False
gf.ENABLE_TAGS = False
gf.ENABLE_HYBRID = False
print("Running with base Markov model")
metric_base = run()

print("Finished all runs")


print("Running t-tests")

for i in range(len(notes)):
    t_test_vector = [metric_full[i], metric_tags[i], metric_emphasis[i], metric_base[i]]
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
