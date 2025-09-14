import json
import sys

import numpy as np

name = sys.argv[1] if len(sys.argv) > 1 else "amrit"

RAGA_DATA_PATH = f"./raga_data_{name}/{name}.json"
TAG_TPM_PATH = f"./model_data_{name}/tpm_tags.npy"
TAG_PICKLE_PATH = f"./model_data_{name}/tag_tpms.pkl"

raga_data = json.load(open(RAGA_DATA_PATH))

new_tags = raga_data["new_tags"]
old_tags = raga_data["tags"]

states = [i.split("-")[0] for i in new_tags]
unique_states = []
for state in states:
    if state not in unique_states:
        unique_states.append(state)

states = unique_states[:]
print(states)


tpm = np.zeros((len(states), len(states)))

for i, tag in enumerate(new_tags[:-1]):
    tpm[states.index(tag.split("-")[0])][
        states.index(new_tags[i + 1].split("-")[0])
    ] += 1

# for i, tag in enumerate(old_tags[:-1]):
#     tpm[states.index(tag.split("-")[0])][states.index(tag.split("-")[0])] += 1

for i, row in enumerate(tpm):
    tpm[i][i] *= 20 / 75
    if np.sum(row) != 0:
        tpm[i] /= np.sum(row)

initial = "I S"

np.save(TAG_TPM_PATH, tpm)

gen = [initial]

i = 0
while len(gen) < 25:
    gen = [initial]
    while not gen[-1].startswith("F"):
        try:
            next = np.random.choice(states, p=tpm[states.index(gen[-1])])
        except:
            print(tpm[states.index(gen[-1])])
            continue

        gen.append(next)
        i += 1
        idx = states.index(next)
        tpm[idx][idx] *= np.random.normal(0.9, 0.1)
        if np.sum(tpm[idx]) != 0:
            tpm[idx] /= np.sum(tpm[idx])

    print(len(gen))

print("\n".join(gen))


allNotes = raga_data["notes"] + ["|"]

tag_tpms = {
    t.split("-")[0]: np.zeros((len(allNotes), len(allNotes))) for t in set(new_tags)
}

for i, tag in enumerate(new_tags):
    phrase = raga_data["new_phrases"][i].split(" ")
    # print(tag, phrase)
    tag = tag.split("-")[0]
    for j, token in enumerate(phrase[:-1]):
        note = token[2:]
        tf = allNotes.index(note)
        tt = allNotes.index(phrase[j + 1][2:])
        tag_tpms[tag][tf][tt] += 1

    tag_tpms[tag][-1][allNotes.index(phrase[0][2:])] += 1
    tag_tpms[tag][allNotes.index(phrase[-1][2:])][-1] += 1

    for i, row in enumerate(tag_tpms[tag]):
        if np.sum(row) != 0:
            tag_tpms[tag][i] /= np.sum(row)

import pickle as pkl

tags_to_time = {}

for tag in raga_data["new_tags"][::-1]:
    t, i = tag.split("-")
    i = int(i)
    tags_to_time[t] = i

# print(tag_tpms.keys())

obj = {"tag_tpms": tag_tpms, "unique_tags": states, "tags_to_time": tags_to_time}

pkl.dump(obj, open(TAG_PICKLE_PATH, "wb"))

import plotly.graph_objects as go

fig = go.Figure(
    data=go.Heatmap(
        z=list(tag_tpms.values())[0],
        colorscale="hot",
        x=allNotes,
        y=allNotes,
    ),
)


steps = []
for tag in states:
    step = dict(method="restyle", args=["z", [tag_tpms[tag]]], label=tag)
    steps.append(step)

    hovertemplate = "From: %{y}<br>To: %{x}<br>P: %{z}"
    fig.data[0].hovertemplate = hovertemplate


sliders = [dict(active=0, currentvalue={"prefix": "Tag: "}, pad={"t": 50}, steps=steps)]

fig.update_layout(sliders=sliders, template="plotly_dark")

fig.update_layout(
    xaxis=go.layout.XAxis(
        title="Next Note",
        tickvals=list(range(len(allNotes))),
        ticktext=allNotes,
    ),
    yaxis=go.layout.YAxis(
        title="Current Note",
        tickvals=list(range(len(allNotes))),
        ticktext=allNotes,
    ),
)

fig.show()
