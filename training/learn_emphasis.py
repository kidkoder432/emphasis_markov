import json
import numpy as np
from pprint import pprint
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle as pkl
from scipy.ndimage import median_filter
from scipy.interpolate import InterpolatedUnivariateSpline, PchipInterpolator

amrit = json.load(open("amrit.json"))

emphasis_table = np.zeros((len(amrit["new_phrases"]), len(amrit["notes"])))

for i, phrase in enumerate(amrit["new_phrases"]):

    phrase = [(p[0], p[1:]) for p in phrase.split(" ")]
    totalEmphasis = sum([int(p[0]) for p in phrase])
    phraseNotes = [p[1] for p in phrase]
    for note in phrase:
        e = int(note[0])
        name = note[1]
        count = phraseNotes.count(name)
        emphasis_table[i][amrit["notes"].index(name)] += e ** (5 / 3)
        print("".join(note), end=" ")

        if phrase[::-1].index(note) == 0:
            emphasis_table[i][amrit["notes"].index(name)] += 1.5 * e ** (5 / 3)

        # emphasis_table[i] **= (2/3)

    emphasis_table[i] /= sum(emphasis_table[i])

    print()


# Convert the emphasis_table into a DataFrame with column titles as notes
emphasis_df = pd.DataFrame(emphasis_table, columns=amrit["notes"])

t = np.linspace(0, len(amrit["new_phrases"]) - 1, 2000)
splines = []
for c in emphasis_df.columns:
    splines.append(
        # InterpolatedUnivariateSpline(
        #     np.arange(len(amrit["new_phrases"])), emphasis_df[c], k=2
        # )
        PchipInterpolator(
            np.arange(len(amrit["new_phrases"])), emphasis_df[c], extrapolate=False
        )
    )

print(emphasis_df)


def adaptive_moving_average(signal, min_window=50, max_window=200):
    # Compute local slope (rate of change)
    slope = np.abs(np.gradient(signal))

    # Normalize slope to [0,1] range
    slope_norm = slope / np.max(slope) if np.max(slope) != 0 else slope

    # Scale window size based on slope
    window_sizes = (min_window + (max_window - min_window) * (1 - slope_norm)).astype(
        int
    )

    # Ensure window sizes are odd for symmetry
    window_sizes += window_sizes % 2 == 0

    # Apply adaptive convolution
    smoothed_signal = np.zeros_like(signal, dtype=np.float64)
    for i in range(len(signal)):
        k = window_sizes[i]  # Get adaptive window size
        kernel = np.ones(k) / k  # Uniform kernel
        start = max(0, i - k // 2)
        end = min(len(signal), i + k // 2 + 1)
        smoothed_signal[i] = np.convolve(signal[start:end], kernel, mode="valid").mean()

    return smoothed_signal

def lpf(signal, a):
    a = 1 - a
    y = np.zeros_like(signal)
    y[0] = signal[0]
    for t in range(1, len(signal)):
        y[t] = a * signal[t] + (1 - a) * y[t - 1]
    return y

from scipy.signal import savgol_filter

def savgol(y, window_size):
    y = np.log(y + 1e-5) 
    return np.exp(savgol_filter(y, window_size, 3))

fig = go.Figure()
convolved_splines = []
for spline, name in zip(splines, emphasis_df.columns):

    # y = np.convolve(spline(t), np.ones(200) / 200, mode="same")
    y = median_filter(spline(t), size=50)
    # y = lpf(spline(t), 0.9)  # spline(t)
    # y = savgol(spline(t), 100)
    # convolved_splines.append(InterpolatedUnivariateSpline(t, y, k=2))
    convolved_splines.append(PchipInterpolator(t, y, extrapolate=False))
    fig.add_scatter(
        x=t,
        y=y,
        mode="lines",
        name=name,
    )

fig.update_layout(
    template="plotly_dark",
    title="Emphasis Table",
    xaxis_title="T",
    yaxis_title="Emphasis Level",
)

maxe_idx = np.array([np.argmax([spline(ti) for spline in convolved_splines]) for ti in t])
max_emphases = np.array([max([spline(ti) for spline in convolved_splines]) for ti in t])

fig.add_scatter(
    x=t, y=max_emphases, mode="lines", name="Max Emphasis", line={"color": "white"}
)

pkl.dump({
    "splines": splines,
    "convolved_splines": convolved_splines
}, open("splines.pkl", "wb"))

fig.show()
