import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

import logging
logger = logging.getLogger(__name__)

# ==============================================================================
# PART 1: STATIC FIGURE GENERATION
# This function creates a single, non-interactive figure.
# You will call this directly to export plots for your paper.
# ==============================================================================
def create_static_figure(data, row_idx, swars_list):
    """
    Generates a single, static Plotly figure for a specific phrase and TPM row.
    """
    emphasis_labels = swars_list[:-1]
    tpm_labels = swars_list
    from_note_label = swars_list[row_idx]

    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.4, 0.4, 0.1],
        subplot_titles=(
            f"<b>Before:</b> Original Probabilities (From: {from_note_label})",
            f"<b>After:</b> Final Emphasized & Normalized Probabilities",
            "<b>Cause:</b> Emphasis Vector",
        ),
        vertical_spacing=0.15,
    )

    # (a) Original TPM Bar Chart
    fig.add_trace(
        go.Bar(
            x=tpm_labels,
            y=data["original_tpm"][row_idx, :],
            name="Original Prob.",
            marker_color="royalblue",
        ),
        row=1,
        col=1,
    )

    # (b) Emphasized TPM Bar Chart
    fig.add_trace(
        go.Bar(
            x=tpm_labels,
            y=data["emphasized_tpm"][row_idx, :],
            name="Final Prob.",
            marker_color="lightseagreen",
        ),
        row=2,
        col=1,
    )

    # (c) Emphasis Vector Heatmap
    fig.add_trace(
        go.Heatmap(
            z=np.array([data["emphasis"]]),
            x=emphasis_labels,
            y=["Emphasis"],
            coloraxis="coloraxis",
            showscale=False,
        ),
        row=3,
        col=1,
    )

    # --- Layout for a clean, static plot ---
    fig.update_layout(
        height=700,
        width=1000,
        title_text=f"Emphasis System Mechanism (From Note: {from_note_label})",
        title_x=0.5,
        template="plotly_dark",
        margin=dict(t=100, b=50),
        yaxis1=dict(range=[0, 1], title="Probability"),
        yaxis2=dict(range=[0, 1], title="Probability"),
        xaxis3=dict(tickangle=-45),
        coloraxis=dict(colorscale="Viridis", cmin=0, cmax=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ==============================================================================
# PART 2: INTERACTIVE DASHBOARD APPLICATION
# This section wraps the static function in a web app for exploration.
# ==============================================================================
def create_visualization(viz_tables, swars_list):
    """
    Launches a Dash web application to interactively explore the visualization data.
    """
    app = dash.Dash(__name__)

    # --- App Layout ---
    app.layout = html.Div(
        style={"backgroundColor": "#111111", "color": "#7FDBFF", "padding": "20px"},
        children=[
            html.H1(
                "Interactive Emphasis System Inspector", style={"textAlign": "center"}
            ),
            dcc.Graph(id="tpm-graph"),
            html.Div(
                [
                    html.Label("Phrase Selector:"),
                    dcc.Slider(
                        id="phrase-slider",
                        min=0,
                        max=len(viz_tables) - 1,
                        value=0,
                        marks={i: str(i) for i in range(len(viz_tables))},
                        step=1,
                    ),
                ],
                style={"padding": "20px 50px"},
            ),
            html.Div(
                [
                    html.Label("'From' Note Selector:"),
                    dcc.Slider(
                        id="note-slider",
                        min=0,
                        max=len(swars_list) - 1,
                        value=0,
                        marks={i: note for i, note in enumerate(swars_list)},
                        step=1,
                    ),
                ],
                style={"padding": "20px 50px"},
            ),
        ],
    )

    # --- Callback to connect sliders to the graph ---
    @app.callback(
        Output("tpm-graph", "figure"),
        [Input("phrase-slider", "value"), Input("note-slider", "value")],
    )
    def update_graph(selected_phrase_idx, selected_note_idx):
        # When a slider changes, this function re-runs create_static_figure
        selected_data = viz_tables[selected_phrase_idx]
        return create_static_figure(selected_data, selected_note_idx, swars_list)

    # --- Run the App ---
    app.run(debug=True)


def create_tpm_dashboard(viz_tables, swars_list):
    """
    Creates an animated Plotly visualization of TPMs and Emphasis,
    using a specific visual style for the figures.
    """
    #
    # --- This section (Error Handling & Data Prep) is preserved from your original function ---
    #
    if not viz_tables:
        logger.warning("No visualization data available to create a plot.")
        return

    logger.info("Creating Plotly visualization...")

    num_phrases_generated = len(viz_tables)

    try:
        # Find the first valid frame
        initial_data = next(
            (
                d
                for d in viz_tables
                if d
                and all(k in d for k in ["emphasized_tpm", "original_tpm", "emphasis"])
                and all(
                    np.all(np.isfinite(d[k]))
                    for k in ["emphasized_tpm", "original_tpm", "emphasis"]
                )
            ),
            None,
        )

        if not initial_data:
            logger.error("No valid data found to initialize visualization.")
            return

        emphasis_labels = swars_list[:-1]
        tpm_labels = swars_list

        figs = make_subplots(
            rows=1,
            cols=3,
            column_widths=[0.45, 0.45, 0.1],  # reduce Emphasis width
            subplot_titles=(
                "Transition Probability Matrix",
                "TPM Original",
                "Emphasis Table",
            ),
            horizontal_spacing=0.05,
        )

        # Add initial heatmaps with coloraxis instead of individual colorbars
        figs.add_trace(
            go.Heatmap(
                z=initial_data["emphasized_tpm"],
                x=tpm_labels,
                y=swars_list,
                coloraxis="coloraxis1",
                xaxis="x1",
                yaxis="y1",
                name="Emphasized TPM",
            ),
            row=1,
            col=1,
        )

        figs.add_trace(
            go.Heatmap(
                z=initial_data["original_tpm"],
                x=tpm_labels,
                y=swars_list,
                coloraxis="coloraxis2",
                xaxis="x2",
                yaxis="y2",
                name="Original TPM",
            ),
            row=1,
            col=2,
        )

        figs.add_trace(
            go.Heatmap(
                z=initial_data["emphasis"].reshape(-1, 1),
                x=["Emphasis"],
                y=emphasis_labels,
                coloraxis="coloraxis3",
                xaxis="x3",
                yaxis="y3",
                name="Emphasis",
            ),
            row=1,
            col=3,
        )

        # Create frames
        frames = []
        valid_indices = []
        for i, data in enumerate(viz_tables):
            if not (
                data
                and all(
                    k in data for k in ["emphasized_tpm", "original_tpm", "emphasis"]
                )
                and all(
                    np.all(np.isfinite(data[k]))
                    for k in ["emphasized_tpm", "original_tpm", "emphasis"]
                )
            ):
                logger.debug(f"Skipping frame {i} due to missing or invalid data.")
                continue

            frames.append(
                go.Frame(
                    data=[
                        go.Heatmap(
                            z=data["emphasized_tpm"],
                            coloraxis="coloraxis1",
                            xaxis="x1",
                            yaxis="y1",
                        ),
                        go.Heatmap(
                            z=data["original_tpm"],
                            coloraxis="coloraxis2",
                            xaxis="x2",
                            yaxis="y2",
                        ),
                        go.Heatmap(
                            z=data["emphasis"].reshape(-1, 1),
                            coloraxis="coloraxis3",
                            xaxis="x3",
                            yaxis="y3",
                        ),
                    ],
                    name=f"Phrase {i}",
                    traces=[0, 1, 2],
                )
            )
            valid_indices.append(i)

        if not frames:
            logger.error("No valid frames to animate.")
            return

        # Slider steps
        slider_steps = [
            {
                "args": [
                    [f"Phrase {i}"],
                    {
                        "frame": {"duration": 300, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 50},
                    },
                ],
                "label": str(i),
                "method": "animate",
            }
            for i in valid_indices
        ]

        # Layout update
        figs.update_layout(
            title_text=f"TPM Evolution & Emphasis (Phrases 0–{num_phrases_generated - 1})",
            title_x=0.5,
            margin=dict(t=120, b=100, l=80, r=60),
            sliders=[
                {
                    "active": 0,
                    "currentvalue": {"prefix": "Phrase: "},
                    "pad": {"t": 50},
                    "steps": slider_steps,
                }
            ],
            xaxis1=dict(title="To Swar/State", tickangle=-45),
            yaxis1=dict(title="From Swar"),
            xaxis2=dict(title="To Swar/State", tickangle=-45),
            yaxis2=dict(title=""),
            xaxis3=dict(title=""),
            yaxis3=dict(title="Swar"),
            hovermode="closest",
            template="plotly_dark",
            # Explicit coloraxis layout
            coloraxis1=dict(colorscale="hot", cmin=0, cmax=1),
            coloraxis2=dict(colorscale="hot", cmin=0, cmax=1),
            coloraxis3=dict(
                colorscale="hot",
                cmin=0,
                cmax=1,
            ),
        )

        figs.frames = frames
        figs.show()
        logger.info("Visualization successfully shown.")

    except Exception as e:
        logger.exception(f"Unexpected error during visualization: {e}")
