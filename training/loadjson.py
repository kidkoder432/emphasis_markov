import json

raga_data = json.load(open("./raga_data_jog/jog.json"))
phrases = str(open("./raga_data_jog/jog.txt").read()).splitlines()
new_phrases = str(open("./raga_data_jog/jog.txt").read()).splitlines()

new_phrases, tags = zip(*[(new_phrase[2:], new_phrase[0]) for new_phrase in new_phrases])
old_phrases, old_tags = zip(*[(phrase[2:], phrase[0]) for phrase in phrases])

raga_data["new_phrases"] = new_phrases
raga_data["tags"] = tags

raga_data["old_phrases"] = old_phrases
raga_data["old_tags"] = old_tags

with open("./raga_data_jog/jog.json", "w") as outfile:
    json.dump(raga_data, outfile, indent=4)
