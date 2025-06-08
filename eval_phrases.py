import pickle as pkl
import numpy as np
import json
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
import pandas as pd
import plotly.graph_objects as go

with open("splines.pkl", "rb") as f:
    splines_ref, convolved_splines_ref = pkl.load(f).values()

with open("phrases.txt", "r") as f:
    phrases = f.read().splitlines()


def savgol(x, window_size):
    x = np.log(x + 1e-5)
    return np.exp(savgol_filter(x, window_size, 3))


amrit = json.load(open("amrit.json"))

emphasis_table = np.zeros((len(phrases), len(amrit["notes"])))

for i, phrase in enumerate(phrases):

    phrase = [(p.split("-")[0], p.split("-")[1]) for p in phrase.split(" ")]
    totalEmphasis = sum([float(p[0]) for p in phrase])
    phraseNotes = [p[1] for p in phrase]
    for note in phrase:
        e = float(note[0])
        name = note[1]
        count = phraseNotes.count(name)
        emphasis_table[i][amrit["notes"].index(name)] += e ** (5 / 3)
        print("".join(note), end=" ")

        if phrase[::-1].index(note) == 0:
            emphasis_table[i][amrit["notes"].index(name)] *= 1.2

        # emphasis_table[i] **= (2/3)

    emphasis_table[i] /= sum(emphasis_table[i])

    print()


# Convert the emphasis_table into a DataFrame with column titles as notes
emphasis_df = pd.DataFrame(emphasis_table, columns=amrit["notes"])

t = np.linspace(0, len(phrases) - 1, 2000)
splines = []
for c in emphasis_df.columns:
    splines.append(
        # InterpolatedUnivariateSpline(
        #     np.arange(len(amrit["new_phrases"])), emphasis_df[c], k=2
        # )
        PchipInterpolator(np.arange(len(phrases)), emphasis_df[c], extrapolate=False)
    )

print(emphasis_df)

fig = go.Figure()
convolved_splines = []
for spline, name in zip(splines, emphasis_df.columns):

    y = savgol(spline(t), 33)
    # convolved_splines.append(InterpolatedUnivariateSpline(t, y, k=2))
    convolved_splines.append(PchipInterpolator(t, y, extrapolate=False))
    fig.add_scatter(
        x=t,
        # y=spline(t),
        y=y,
        mode="lines",
        name=name,
    )

fig_ref = go.Figure()
for spline_ref, name in zip(convolved_splines_ref, emphasis_df.columns):

    t_ref = np.linspace(0, len(amrit["new_phrases"]) - 1, 2000)
    y_ref = spline_ref(t_ref)
    fig_ref.add_scatter(
        x=t,
        y=y_ref,
        mode="lines",
        name=name,
    )

fig_ref.update_layout(
    template="plotly_dark",
    title="Reference Splines",
    xaxis_title="T",
    yaxis_title="Emphasis Level",
)

fig_ref.show()


fig.update_layout(
    template="plotly_dark",
    title="Emphasis Table",
    xaxis_title="T",
    yaxis_title="Emphasis Level",
)

pkl.dump(
    {"splines": splines, "convolved_splines": convolved_splines},
    open("splines_eval.pkl", "wb"),
)

fig.show()


from sktime.distances import dtw_distance

distances = {}
for i in range(len(emphasis_df.columns)):
    distances[emphasis_df.columns[i]] = dtw_distance(
        convolved_splines[i](t), convolved_splines_ref[i](t_ref)
    ) / len(phrases)

print(json.dumps(distances, indent=4))
