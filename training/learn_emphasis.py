from email.policy import default
import json
import pickle as pkl
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter

config = {
    "toImageButtonOptions": {
        "format": "png",  # one of png, svg, jpeg, webp
        "scale": 4,  # Multiply resolution by this factor
    }
}

def calculate_emphasis_df(raga_data):
    """
    Calculates an emphasis table for musical phrases from a given data dictionary.

    The emphasis table is a DataFrame where rows represent phrases and columns
    represent notes. The value at each cell indicates the relative emphasis of a
    specific note within a phrase.

    Args:
        amrit_data (dict): A dictionary containing 'new_phrases' and 'notes'.
                           'new_phrases' is a list of strings, where each string
                           is a space-separated list of notes with emphasis values (e.g., "1.5SA 2RE").
                           'notes' is a list of note names.

    Returns:
        pd.DataFrame: A DataFrame representing the emphasis table.
    """
    emphasis_table = np.zeros((len(raga_data["phrases"]), len(raga_data["notes"])))

    for i, phrase_str in enumerate(raga_data["phrases"]):
        # Split the phrase into individual note-emphasis pairs
        phrase_parts = phrase_str.split(" ")

        # Prepare a list of just the note names for easy counting
        # phrase_notes = [part[1:] for part in phrase_parts]

        # Calculate emphasis for each note in the current phrase
        for part in phrase_parts:
            # Handle potential empty strings from splitting
            if not part:
                continue

            # Parse the emphasis value and note name
            try:
                part = part.split("-")
                emphasis_value = float(part[0])
                note_name = part[1]
            except (ValueError, IndexError):
                print(
                    f"Warning: Could not parse '{part}' in phrase '{phrase_str}'. Skipping."
                )
                continue

            # Update the emphasis table with a power law for non-linear emphasis
            emphasis_table[i][
                raga_data["notes"].index(note_name)
            ] += emphasis_value ** (5 / 3)

            # Apply a special emphasis boost for the last note of a phrase
            # NOTE: phrase_parts is already a list of strings, no need to reverse and index
            if part == phrase_parts[-1]:
                emphasis_table[i][raga_data["notes"].index(note_name)] *= 2.5

        # Normalize the row so that the total emphasis for each phrase sums to 1
        row_sum = np.sum(emphasis_table[i])
        if row_sum > 0:
            emphasis_table[i] /= row_sum

    # Convert the numpy array into a DataFrame with note names as columns
    emphasis_df = pd.DataFrame(emphasis_table, columns=raga_data["notes"])
    return emphasis_df


def generate_splines(emphasis_df):
    # Create a dense time vector for smooth plotting
    t = np.linspace(0, len(emphasis_df) - 1, 2000)

    # Generate PchipInterpolator splines for each note column
    splines = []
    for c in emphasis_df.columns:
        spline = PchipInterpolator(
            np.arange(len(emphasis_df)), emphasis_df[c], extrapolate=False
        )
        splines.append(spline)

    # Initialize a plotly figure for the plot
    convolved_splines = []

    # Process and plot each note's emphasis curve
    for spline, name in zip(splines, emphasis_df.columns):
        # Apply a median filter for smoothing
        y = median_filter(spline(t), size=35)

        # Store the smoothed spline for later use
        convolved_splines.append(PchipInterpolator(t, y, extrapolate=False))

    return t, splines, convolved_splines


def generate_plots_and_save_splines(
    emphasis_df, t, splines, convolved_splines, output_path
):
    """
    Generates an emphasis plot with splines and saves the spline models.

    Args:
        emphasis_df (pd.DataFrame): The DataFrame containing emphasis values.
        output_path (str): The file path to save the pickled splines.
    """

    # Create a dense time vector for smooth plotting
    t = np.linspace(0, len(emphasis_df) - 1, 2000)

    # Initialize a plotly figure for the plot
    fig = go.Figure()

    # Define your styles
    # 'solid' is default. 'dash' and 'dot' are good secondary options.
    line_styles = ["solid", "dash", "dot"]

    colors = px.colors.qualitative.Dark2

    # Use enumerate to get the index 'i' for each trace
    for i, (spline, name) in enumerate(zip(convolved_splines, emphasis_df.columns)):

        if sum(spline(t)) == 0:
            continue

        # LOGIC: Switch style every 8 traces
        # Traces 0-7: Solid
        # Traces 8-15: Dash
        # Traces 16+: Dot (if you had them)
        current_style = line_styles[(i // 8) % len(line_styles)]

        fig.add_scatter(
            x=t,
            y=spline(t),
            mode="lines",
            name=name,
            line={
                "color": colors[i % len(colors)],  # Safety modulo
                "dash": current_style,  # Apply the style
            },
        )

    # Add the "Max Emphasis" line to the plot
    # max_emphases = np.array(
    #     [max([spline(ti) for spline in convolved_splines]) for ti in t]
    # )
    # fig.add_scatter(
    #     x=t, y=max_emphases, mode="lines", name="Max Emphasis", line={"color": "white"}
    # )

    # Update plot layout and title
    fig.update_layout(
        template="plotly_white",
        title="Emphasis Table",
        xaxis_title="Phrase #",
        yaxis_title="Emphasis Level",
        # 1. MAKE THE CANVAS SMALL (Simulates the column width)
        width=1000,
        height=600,
        # 2. MAKE THE TEXT HUGE RELATIVE TO THE CANVAS
        font=dict(
            family="Verdana",
            size=22,  # This looks massive on screen, but perfect in the paper
            color="black",
        ),
    )

    # Save the splines to a pickle file for later use
    pkl.dump(
        {"t": t, "splines": splines, "convolved_splines": convolved_splines},
        open(output_path, "wb"),
    )

    # Display the final plot
    fig.show(config=config)

    names = emphasis_df.columns.tolist()

    fig2 = go.Figure()
    max_emp = {note: [None] * len(t) for note in emphasis_df.columns}
    for i, time in enumerate(t):
        vec = [spline(time) for spline in convolved_splines]
        name = emphasis_df.columns[np.argmax(vec)]
        max_emp[name][i] = time

    colors = px.colors.qualitative.Light24

    for name, times in max_emp.items():
        if all(time is None for time in times):
            continue
        fig2.add_scatter(
            x=times,
            y=[name] * len(times),
            mode="lines",
            name=name,
            connectgaps=False,
            line={"color": "steelblue"},
            
        )

    fig2.update_layout(
        template="plotly_white",
        title="Maximal Emphasis Regions",
        xaxis_title="Phrase #",
        yaxis_title="Note",
        showlegend=False,
         # 1. MAKE THE CANVAS SMALL (Simulates the column width)
        width=1200,
        height=600,
        # 2. MAKE THE TEXT HUGE RELATIVE TO THE CANVAS
        font=dict(
            family="Verdana",
            size=22,  # This looks massive on screen, but perfect in the paper
            color="black",
        ),
    )

    for i in range(1, int(max(t))):
        fig2.add_shape(
            go.layout.Shape(
                type="line",
                x0=i,
                x1=i,
                y0=0,
                y1=len(names),
                line={"color": "lightgrey", "width": 0.5}
            )
        )

    fig2.show(config=config)


if __name__ == "__main__":
    # --- Main script execution starts here ---

    # Get the raga name from command line arguments if provided
    raga_name = sys.argv[1] if len(sys.argv) > 1 else "amrit"

    # Define the path to your data file
    data_file_path = f"./raga_data_{raga_name}/{raga_name}.json"
    spline_output_path = f"./model_data_{raga_name}/splines.pkl"

    # Load the data from the JSON file
    try:
        raga_data = json.load(open(data_file_path))
    except FileNotFoundError:
        print(
            f"Error: The file '{data_file_path}' was not found. Please ensure the path is correct."
        )
        exit()

    data = {}
    data["notes"] = raga_data["notes"]
    data["phrases"] = raga_data["new_phrases"]

    # Calculate the emphasis DataFrame using the reusable function
    emphasis_df = calculate_emphasis_df(data)
    print("Calculated Emphasis DataFrame:")
    print(emphasis_df)

    # Generate the plots and save the spline models using the reusable function
    generate_plots_and_save_splines(
        emphasis_df, *generate_splines(emphasis_df), spline_output_path
    )

    print(f"\nSpline models saved to '{spline_output_path}'")
