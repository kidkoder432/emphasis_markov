from eval_full import get_dataset_vistaar, generate_vistaars, gen

import generate_full as gf

from collections import Counter

import plotly.graph_objects as go

gf.changeRaga("amrit")

NUM_SAMPLES = 10


def getNotes(vistaar):
    notes = []
    for phrase in vistaar:
        # print(phrase)
        for n in phrase.split(" ")[3:]:
            notes.append(n.split("-")[1])
    return notes


def ngrams_set(s, n):
    return {tuple(s[i : i + n]) for i in range(len(s) - n + 1)}


def calcNgrams(v1, v2, n):
    v1 = getNotes(v1)  # Dataset
    v2 = getNotes(v2)  # Generated

    g_dataset = ngrams_set(v1, n)
    g_gen = ngrams_set(v2, n)

    # Avoid division by zero if generation is empty
    if len(g_gen) == 0:
        return 0

    # Precision: What % of the GENERATED unique n-grams are "real" (exist in dataset)?
    intersection = len(g_dataset & g_gen)
    return intersection / len(g_gen)


def eval_recall(dataset_ngrams, all_gen_vistaars, n):
    # 1. Collect ALL unique n-grams from ALL generations
    all_gen_ngrams = set()
    for v in all_gen_vistaars:
        all_gen_ngrams.update(ngrams_set(getNotes(v), n))

    # 2. Check overlap with Dataset
    shared = dataset_ngrams.intersection(all_gen_ngrams)

    # 3. Calculate Percentage of Dataset Covered
    if len(dataset_ngrams) == 0:
        return 0
    return len(shared) / len(dataset_ngrams)


dataset = get_dataset_vistaar()[0]


def eval_ngrams(dataset, gen_vistaars):
    metric = []
    recall_scores = []
    for n in range(1, 21):
        totalNgrams = sum(
            calcNgrams(dataset, gen_vistaars[i], n) for i in range(len(gen_vistaars))
        )
        # print(n, totalNgrams / NUM_SAMPLES)
        metric.append(totalNgrams / len(gen_vistaars))

        d_ngrams = ngrams_set(getNotes(dataset), n)
        score = eval_recall(d_ngrams, gen_vistaars, n)
        recall_scores.append(score)
    return metric, recall_scores


print("Running with emphasis, tags, and hybrid enabled")
gf.ENABLE_EMPHASIS = True
gf.ENABLE_TAGS = True
gf.ENABLE_HYBRID = True
vistaars = generate_vistaars(NUM_SAMPLES)
ngrams_full, recall_scores_full = eval_ngrams(dataset, vistaars)

print("Running with emphasis and tags enabled")
gf.ENABLE_EMPHASIS = True
gf.ENABLE_TAGS = True
gf.ENABLE_HYBRID = False
vistaars = generate_vistaars(NUM_SAMPLES)
ngrams_tags, recall_scores_tags = eval_ngrams(dataset, vistaars)

gen = []
print("Running with emphasis enabled")
gf.ENABLE_EMPHASIS = True
gf.ENABLE_TAGS = False
gf.ENABLE_HYBRID = False
vistaars = generate_vistaars(NUM_SAMPLES, gen=gen)
ngrams_emphasis, recall_scores_emphasis = eval_ngrams(dataset, vistaars)

print("Running with base Markov model")
gf.ENABLE_EMPHASIS = False
gf.ENABLE_TAGS = False
gf.ENABLE_HYBRID = False
vistaars = generate_vistaars(NUM_SAMPLES, gen=gen)
ngrams_base, recall_scores_base = eval_ngrams(dataset, vistaars)

ngrams_dataset, recall_scores_dataset = eval_ngrams(dataset, [dataset])

import csv

with open("overfit.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(
        [
            "Model",
            "Full",
            "Tags",
            "Emphasis",
            "Base",
            "Dataset",
            "Recall Score (Full)",
            "Recall Score (Tags)",
            "Recall Score (Emphasis)",
            "Recall Score (Base)",
            "Recall Score (Dataset)",
        ]
    )

    for row in zip(
        ngrams_full,
        ngrams_tags,
        ngrams_emphasis,
        ngrams_base,
        ngrams_dataset,
        recall_scores_full,
        recall_scores_tags,
        recall_scores_emphasis,
        recall_scores_base,
        recall_scores_dataset,
    ):
        writer.writerow(list(row))

fig = go.Figure(
    data=[
        go.Bar(name="Full", x=list(range(1, 21)), y=ngrams_full),
        go.Bar(name="Tags", x=list(range(1, 21)), y=ngrams_tags),
        go.Bar(name="Emphasis", x=list(range(1, 21)), y=ngrams_emphasis),
        go.Bar(name="Base", x=list(range(1, 21)), y=ngrams_base),
        go.Bar(name="Dataset", x=list(range(1, 21)), y=ngrams_dataset),
    ]
)

fig.update_layout(template="plotly_dark")

fig.show()


# Add this to your plot
fig = go.Figure(
    data=[
        go.Scatter(
            name="Full",
            x=list(range(1, 21)),
            y=recall_scores_full,
            yaxis="y2",
            mode="lines+markers",
        ),
        go.Scatter(
            name="Tags",
            x=list(range(1, 21)),
            y=recall_scores_tags,
            yaxis="y2",
            mode="lines+markers",
        ),
        go.Scatter(
            name="Emphasis",
            x=list(range(1, 21)),
            y=recall_scores_emphasis,
            yaxis="y2",
            mode="lines+markers",
        ),
        go.Scatter(
            name="Base",
            x=list(range(1, 21)),
            y=recall_scores_base,
            yaxis="y2",
            mode="lines+markers",
        ),
        go.Scatter(
            name="Dataset",
            x=list(range(1, 21)),
            y=recall_scores_dataset,
            yaxis="y2",
            mode="lines+markers",
        ),
    ]
)

# You might need a secondary Y-axis since this is 0.0-1.0 while previous counts were 0-400
fig.update_layout(
    yaxis2=dict(title="Coverage %", overlaying="y", side="right", range=[0, 1]),
    template="plotly_dark",
)
fig.show()
