from math import isnan
import pickle as pkl
import numpy as np
import json

from plotly.subplots import make_subplots
import plotly.graph_objects as go

tpm = np.load("tpm.npy")

with open("amrit.json", "r") as f:
    amrit = json.load(f)

SWARS = amrit["notes"] + ["|"]

with open("splines.pkl", "rb") as f:
    splines, convolved_splines = pkl.load(f).values()


# domain is 0-76
def get_emphasis(t, tmin, tmax):
    t_scaled = (t - tmin) / (tmax - tmin) * 74

    out = np.array([spline(t_scaled) for spline in convolved_splines])

    out /= np.sum(out)
    return out


def clip(x, xmin, xmax):
    return max(min(x, xmax), xmin)


startNote = "S"

phrases = []

tables = []

numOfPhrases = 25

for phraseNum in range(numOfPhrases):
    t = clip(phraseNum + np.random.normal(0.1, 0.1), 0, numOfPhrases)

    emphasis = get_emphasis(t, 0, numOfPhrases)

    emphasis = np.where(np.isclose(emphasis, 0, atol=1e-6), 0, emphasis)

    new_tpm = np.zeros((tpm.shape[0], tpm.shape[1] + 1))

    new_tpm[:, -2] = tpm[:, -1]
    new_tpm[-1, :-1] = tpm[-1, :]

    col_emp = np.where(
        np.isclose(emphasis[None, :], 0, atol=5e-3), 0, emphasis[None, :]
    )
    # row_emp = np.where(
    #     np.isclose(emphasis[:, None], 0, atol=5e-3), 0, emphasis[:, None]
    # ) ** (1 / 2)

    # new_tpm[:-1, :-1] = tpm[:-1, :] * row_emp
    new_tpm[:, :-2] = tpm[:, :-1] * col_emp

    new_tpm[:-1, -2] *= col_emp.flatten()

    new_tpm = np.where(np.isclose(new_tpm, 0, atol=1e-6), 0, new_tpm)

    # for i in tpm:
    #     for j in i:
    #         print(round(j, 2), end=" ")
    #     print()

    for row in new_tpm:
        if np.sum(row) < 1:
            row[-1] = 1 - np.sum(row[:-1])

    # probs = new_tpm[-1][:-2] / np.sum(new_tpm[-1][:-2])
    probs = get_emphasis(t, 0, numOfPhrases)

    note = np.random.choice(SWARS[:-1], p=probs)
    while np.sum(new_tpm[SWARS.index(note)][:-1]) == 0:
        note = np.random.choice(SWARS[:-1], p=probs)
    phrase = []

    tables.append([emphasis, new_tpm, tpm])

    T = 1.25
    total_duration = 0.0001
    prev_note = note

    while note != "|" or len(phrase) < 3:
        # print(note, new_tpm[SWARS.index(note)], np.sum(new_tpm[SWARS.index(note)]))

        p = new_tpm[SWARS.index(note)]
        # p[-2] = total_duration / 40

        p /= np.sum(p)
        index = np.random.choice(range(len(new_tpm[SWARS.index(note)])), p=p)

        t = clip(t + np.random.normal(0.2, 0.2), phraseNum, phraseNum + 1)

        # do nothing
        # if we need to retry
        if index >= len(SWARS):

            k = new_tpm[SWARS.index(note)].copy()
            k /= np.sum(k)

            # Apply a small abount of smoothing each time before retrying
            # Expected number of retries = P(retry) / (1 - P(retry))
            # We should make the temperature 2 at the end.

            k **= 1 / T

            k /= np.sum(k)
            print(note, k)
            new_tpm[SWARS.index(note)] = k.copy()

            continue
        else:
            note = SWARS[index]

        # Repetition is illegal
        if note == prev_note:
            continue

        # Normal case
        if note != "|":
            current_duration = 10 * get_emphasis(t, 0, numOfPhrases)[SWARS.index(note)]

            if current_duration <= 0.4:
                continue
            phrase.append((current_duration, note))
            total_duration += current_duration

        prev_note = note
    # Phrase is over
    # print(" ".join([f"{np.ceil(e[0])}{e[1]}" for e in phrase]))

    phrases.append(" ".join([str(t)] + [f"{e[0]:.2f}-{e[1]}" for e in phrase]))
    print("--------")

with open("phrases.txt", "w") as f:
    f.write("\n".join(phrases))

import mido

mid = mido.MidiFile(ticks_per_beat=480)

track = mido.MidiTrack()
mid.tracks.append(track)

swars_json = json.load(open("swars.json"))

swar2midi = swars_json["swar2midi"]

bpm = 100
tonic = 56
ticks = mido.bpm2tempo(bpm)

track.append(mido.MetaMessage("set_tempo", tempo=ticks, time=0))

for phrase in phrases:
    print(phrase)
    for note in phrase.split(" ")[1:]:
        print(note)
        duration, pitch = note.split("-")

        # Add note_on message
        track.append(
            mido.Message(
                "note_on",
                note=swar2midi[pitch] + tonic,
                velocity=64,
                time=10,  # Small time before the note starts
            )
        )

        # Calculate note duration in ticks
        note_duration = int(float(duration) * mid.ticks_per_beat)

        # Add note_off message with appropriate duration
        track.append(
            mido.Message(
                "note_off",
                note=swar2midi[pitch] + tonic,
                velocity=64,
                time=note_duration,
            )
        )

        # Add a small pause after the note-off to avoid overlap
        track.append(
            mido.Message(
                "note_off",
                note=swar2midi[pitch] + tonic,
                velocity=0,  # No velocity for this "pause"
                time=10,  # Small pause time
            )
        )

    track.append(
        mido.Message(
            "note_off",
            velocity=0,
            time=int(mid.ticks_per_beat * np.random.normal(4, 0.3)),
        )
    )

mid.save("amrit.mid")


# Create initial figures
figs = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=("Transition Probability Matrix", "TPM Original", "Emphasis Table"),
)

heatmap1 = go.Heatmap(
    z=tables[0][1],
    colorscale="hot",
    name="Transition Probability Matrix",
    x=SWARS + ["x"],
    y=SWARS,
)
heatmap2 = go.Heatmap(
    z=tables[0][0].reshape(-1, 1),
    colorscale="hot",
    name="Emphasis Table",
    y=SWARS,
    x=["Emphasis"],
)
heatmap3 = go.Heatmap(
    z=tables[0][2],
    colorscale="hot",
    name="Transition Probability Matrix Original",
    x=SWARS,
    y=SWARS,
)

figs.add_trace(heatmap1, row=1, col=1)
figs.add_trace(heatmap3, row=1, col=2)
figs.add_trace(heatmap2, row=1, col=3)

# Define frames for animation
frames = [
    go.Frame(
        data=[
            go.Heatmap(z=tables[i][1], colorscale="hot"),
            go.Heatmap(z=tables[i][2], colorscale="hot"),
            go.Heatmap(z=tables[i][0].reshape(-1, 1), colorscale="hot"),
        ],
        name=f"Phrase {i}",
    )
    for i in range(len(tables))
]

# Define slider steps
slider_steps = [
    {
        "args": [
            [f"Phrase {i}"],
            {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
        ],
        "label": f"Phrase {i}",
        "method": "animate",
    }
    for i in range(len(tables))
]

# Add animation controls
figs.update_layout(
    title="Transition Probability Matrices and Emphasis Table",
    xaxis=dict(
        title="To",
        tickmode="array",
        tickvals=list(range(len(SWARS) + 1)),  # Ensure tick positions match labels
        ticktext=SWARS + ["x"],
    ),
    yaxis=dict(
        title="From",
        tickmode="array",
        tickvals=list(range(len(SWARS))),
        ticktext=SWARS,
    ),
    sliders=[
        {
            "active": 0,
            "currentvalue": {"prefix": "Phrase: "},
            "pad": {"t": 50},
            "steps": slider_steps,
        }
    ],
    template="plotly_dark",
)

# **Manually add frames to the Figure object**
figs.frames = frames

figs.show()
