import json
import numpy as np
import pickle as pkl

DEFAULT_RAGADATA_PATH = "./raga_data/amrit.json"
DEFAULT_TPM_PATH = "./model_data/tpm.npy"
DEFAULT_TAG_TPM_PATH = "./model_data/tpm_tags.npy"
DEFAULT_TAG_PICKLE_PATH = "./model_data/tag_tpms.pkl"

tpm = np.load(DEFAULT_TPM_PATH)
amrit_data = json.load(open(DEFAULT_RAGADATA_PATH))

# --- Tag stuff ---
tags = amrit_data["tags"]
unique_tags = list(set(tags))
note_tags = amrit_data["new_tags"]

tag_tpm = np.load(DEFAULT_TAG_TPM_PATH)

tag_tpm_dict, unique_note_tags, tags_to_time = pkl.load(
    open(DEFAULT_TAG_PICKLE_PATH, "rb")
).values()

lens = []
for k in range(1000):
    i = 0
    initial_state = "I S"
    gen = [initial_state]
    current_time = 0
    while len(gen) < 25:
        gen = [initial_state]
        while not gen[-1].startswith("F"):
            try:
                next = np.random.choice(
                    unique_note_tags, p=tag_tpm[unique_note_tags.index(gen[-1])]
                )
                if tags_to_time[next] < current_time:
                    continue
            except:
                print(tag_tpm[unique_note_tags.index(gen[-1])])
                continue

            gen.append(next)
            i += 1
            current_time = tags_to_time[next]
            idx = unique_note_tags.index(next)
            if np.sum(tag_tpm[idx]) != 0:
                tag_tpm[idx] /= np.sum(tag_tpm[idx])

    lens.append(len(gen))
    print(len(gen))

gen = np.unique(gen).tolist()
print("Generated sequence:", gen)
print("Number of unique tags:", len(gen))
print("Average length of generated sequences:", np.mean(lens))
print("Standard deviation of generated sequences:", np.std(lens))
print("Median length of generated sequences:", np.median(lens))
print("Minimum length of generated sequences:", np.min(lens))
print("Maximum length of generated sequences:", np.max(lens))
