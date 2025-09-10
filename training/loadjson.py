import json
import sys

name = sys.argv[1] if len(sys.argv) > 1 else "amrit"

if name == "amrit":
    RAGA_DATA = "./raga_data_amrit/amrit.json"
    PHRASES = "./raga_data_amrit/amrit copy.txt"
elif name == "jog":
    RAGA_DATA = "./raga_data_jog/jog.json"
    PHRASES = "./raga_data_jog/jog.txt"

raga_data = json.load(open(RAGA_DATA))
phrases = str(open(PHRASES).read()).splitlines()
new_phrases = str(open(PHRASES).read()).splitlines()

new_phrases, tags = zip(
    *[(new_phrase[2:], new_phrase[0]) for new_phrase in new_phrases]
)
old_phrases, old_tags = zip(*[(phrase[2:], phrase[0]) for phrase in phrases])

raga_data["new_phrases"] = new_phrases
raga_data["tags"] = tags

raga_data["old_phrases"] = old_phrases
raga_data["old_tags"] = old_tags

with open(RAGA_DATA, "w") as outfile:
    json.dump(raga_data, outfile, indent=4)
