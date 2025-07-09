from tkinter import SW
import numpy as np
import json

with open("amrit.json", "r") as f:
    amrit = json.load(f)

phrases_file = open("./outputs/phrases_0509_2.txt").read().splitlines()

gen_phrases = []

for line in phrases_file:
    phrase = line.split(" ")[3:]
    phrase_str = " ".join(["x" + phrase.split("-")[1] for phrase in phrase])
    gen_phrases.append(phrase_str)
    
print(gen_phrases[:5])

SWARS = amrit["notes"] + ["|"]

tpm = np.zeros((len(SWARS), len(SWARS)))

phrases = amrit["phrases"] + amrit["new_phrases"] + gen_phrases

important_notes = ["S", "S_", "S^"]

v = amrit["vadi"]
important_notes += [v, v + "_", v + "^"]

sv = amrit["samvadi"]
important_notes += [sv, sv + "_", sv + "^"]

important_notes = list(set(important_notes))

for phrase in phrases:
    phrase = phrase.split(" ")

    phrase = [(p[0], p[1:]) for p in phrase]

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
    aspect="auto",
    color_continuous_scale="turbo",
)

fig.update_xaxes(title="To")
fig.update_yaxes(title="From")

fig.update_layout(template="plotly_dark")
fig.show()

# print(tpm)
np.save("tpm.npy", tpm)
