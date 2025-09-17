import numpy as np
import plotly.graph_objects as go
import pickle as pkl

with open("./model_data_amrit/tag_tpms.pkl", "rb") as f:
    _, unique_note_tags, tag_tpms = pkl.load(f).values()

tag_tpm = np.load("./model_data_amrit/tpm_tags.npy")

fig = go.Figure(data=go.Heatmap(z=tag_tpm, colorscale="hot", x=unique_note_tags, y=unique_note_tags))
fig.update_layout(title="Tag Transition Probability Matrix", template="plotly_dark")
fig.show()
