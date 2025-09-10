import json
import pickle as pkl
import sys

import numpy as np
import scipy as sp

name = sys.argv[1] if len(sys.argv) > 1 else "amrit"

SPLINES_FILE = f"./model_data_{name}/splines.pkl"
RAGA_DATA_FILE = f"./raga_data_{name}/{name}.json"

t, splines, convolved_splines = pkl.load(open(SPLINES_FILE, "rb")).values()

raga_data = json.load(open(RAGA_DATA_FILE))

notes = raga_data["notes"]
note = None
new_tags = []
for i, tag in enumerate(raga_data["tags"]):
    if tag == "I" or tag == "R":
        note = notes[np.argmax([sp(i) for sp in splines])]
    elif tag == "F":
        note = "S"

    if note is not None:
        print(tag, note)

        new_tags.append(f"{tag} {note}-{i}")

raga_data["new_tags"] = new_tags

with open(RAGA_DATA_FILE, "w") as outfile:
    json.dump(raga_data, outfile, indent=4)
