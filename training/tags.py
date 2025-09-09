import json
import numpy as np
import pickle as pkl

t, splines, convolved_splines = pkl.load(open("./model_data_jog/splines.pkl", "rb")).values()

raga_data = json.load(open("./raga_data_jog/jog.json"))

notes = raga_data["notes"]
note = None
new_tags = []
for i, tag in enumerate(raga_data["tags"]):
    if tag == "I" or tag == "R":
        note = notes[np.argmax([sp(i).max() for sp in convolved_splines])]
    elif tag == "F":
        note = "S"

    if note is not None:
        print(tag, note)

        new_tags.append(f"{tag} {note}-{i}")

raga_data["new_tags"] = new_tags

with open("./raga_data_jog/jog.json", "w") as outfile:
    json.dump(raga_data, outfile, indent=4)