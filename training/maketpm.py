from tkinter import SW
import numpy as np
import json

with open("./raga_data_jog/jog.json", "r") as f:
    raga_data = json.load(f)

SWARS = raga_data["notes"] + ["|"]

tpm = np.zeros((len(SWARS), len(SWARS)))

phrases = raga_data["old_phrases"] + raga_data["new_phrases"]

important_notes = ["S", "S_", "S^"]

v = raga_data["vadi"]
important_notes += [v, v + "_", v + "^"]

sv = raga_data["samvadi"]
important_notes += [sv, sv + "_", sv + "^"]

important_notes = list(set(important_notes))

for phrase in phrases:
    phrase = phrase.split(" ")

    phrase = [(p[0], p[2:]) for p in phrase]

    for i in range(len(phrase) - 1):
        tpm[SWARS.index(phrase[i][1]), SWARS.index(phrase[i + 1][1])] += 1.0

    tpm[-1][SWARS.index(phrase[0][1])] += 1.0
    tpm[SWARS.index(phrase[-1][1])][-1] += 1.0

for i in important_notes:
    tpm[SWARS.index(i)] *= 1.2
    tpm.T[SWARS.index(i)] *= 1.2

print(np.sum(tpm), len(phrases))
for i in range(len(SWARS)):
    if np.sum(tpm[i]) > 0:
        tpm[i] /= np.sum(tpm[i])

import plotly.express as px

fig = px.imshow(
    tpm,
    labels=dict(x="To", y="From", color="Transition Probability"),
    x=SWARS,
    y=SWARS,
    color_continuous_scale="hot",
    aspect="equal",
)

fig.update_xaxes(title="To")
fig.update_yaxes(title="From")

fig.update_layout(template="plotly_dark")
fig.show()

# print(tpm)
np.save("./model_data_jog/tpm.npy", tpm)
