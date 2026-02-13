import os
import re
import json
import sys


def delta(l):
    t = []
    for i in range(1, len(l)):
        print(l[i], l[i - 1], l[i] - l[i - 1])
        t.append(l[i] - l[i - 1])
    return t


s2m = json.load(open("swars.json"))["swar2midi"]


def get_notes(fn):
    notes = []

    file_path = os.path.join(fn)
    with open(file_path, "r") as f:
        content = f.read()
        for line in content.splitlines():
            match = re.findall(r"(\d(\.\d\d)?)-(..?)\s", line)
            for m in match:
                notes.append(s2m[m[2]])
                print(m)
    return notes


notes = get_notes(sys.argv[1])

print(max(delta(notes)), min(delta(notes)))
