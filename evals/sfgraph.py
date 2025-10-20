import openpyxl
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

name = input("Name: ")

wb = openpyxl.load_workbook(f"./1019_evals_{name}.xlsx")
sheet = wb.sheetnames[-1]
ws = wb[sheet]


cells = ws["B3:E17"]
std = ws["B22:E36"]

cells = [[i.value for i in j] for j in cells]
std = [[i.value for i in j] for j in std]

std = pd.DataFrame(std).transpose()
cells = pd.DataFrame(cells).transpose()

cells.loc[cells.shape[0]] = 0
std.loc[std.shape[0]] = 0

labels = json.load(open(f"../raga_data_{name}/{name}.json"))["notes"]
models = ["Full Model", "Emphasis + Tags", "Emphasis-Only", "Baseline", "Dataset"]

fig = go.Figure()
colors = px.colors.qualitative.Dark2

for i, model in enumerate(models):
    y = cells.iloc[i, :].to_numpy(dtype=float)
    err = std.iloc[i, :].to_numpy(dtype=float)
    color = colors[i % len(colors)]

    # main line
    fig.add_scatter(
        x=labels,
        y=y,
        mode="lines+markers",
        error_y=dict(array=err, visible=True),
        name=model,
    )

    # fig.add_bar(
    #     x=labels,
    #     y=y,
    #     marker_color=color,
    #     error_y=dict(type="data", array=err),
    #     name=model,
    # )


fig.update_layout(
    xaxis_title="Note", yaxis_title="Euclidean Distancex", template="plotly_dark"
)

fig.show()
