import json

amrit = json.load(open("amrit.json"))
phrases = str(open("./amrit.txt").read()).splitlines()
new_phrases = str(open("./amrit copy.txt").read()).splitlines()

new_phrases, tags = zip(*[(new_phrase[2:], new_phrase[0]) for new_phrase in new_phrases])
old_phrases, old_tags = zip(*[(phrase[2:], phrase[0]) for phrase in phrases])

amrit["new_phrases"] = new_phrases
amrit["tags"] = tags

amrit["old_phrases"] = old_phrases
amrit["old_tags"] = old_tags

with open("amrit.json", "w") as outfile:
    json.dump(amrit, outfile, indent=4)
