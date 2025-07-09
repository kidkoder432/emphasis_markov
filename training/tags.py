import json
import numpy as np
import pickle as pkl

splines, convolved_splines = pkl.load(open("splines.pkl", "rb")).values()

amrit = json.load(open("amrit.json"))

notes = amrit["notes"]
note = None
new_tags = []
for i, tag in enumerate(amrit["tags"]):
    if tag == "I" or tag == "R":
        note = notes[np.argmax([sp(i).max() for sp in convolved_splines])]
    elif tag == "F":
        note = "S"

    if note is not None:
        print(tag, note)

        new_tags.append(f"{tag} {note}-{i}")

amrit["new_tags"] = new_tags

with open("amrit.json", "w") as outfile:
    json.dump(amrit, outfile, indent=4)