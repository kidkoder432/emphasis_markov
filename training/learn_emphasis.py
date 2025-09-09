import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle as pkl
from scipy.ndimage import median_filter
from scipy.interpolate import PchipInterpolator


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
    emphasis_table = np.zeros(
        (len(raga_data["phrases"]), len(raga_data["notes"]))
    )

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

def generate_plots_and_save_splines(emphasis_df, t, splines, convolved_splines, output_path):
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

    # Process and plot each note's emphasis curve
    for spline, name in zip(convolved_splines, emphasis_df.columns):
        
        # Add the smoothed curve to the plot
        fig.add_scatter(
            x=t,
            y=spline(t),
            mode="lines",
            name=name,
        )

    # Add the "Max Emphasis" line to the plot
    max_emphases = np.array(
        [max([spline(ti) for spline in convolved_splines]) for ti in t]
    )
    fig.add_scatter(
        x=t, y=max_emphases, mode="lines", name="Max Emphasis", line={"color": "white"}
    )

    # Update plot layout and title
    fig.update_layout(
        template="plotly_dark",
        title="Emphasis Table",
        xaxis_title="T",
        yaxis_title="Emphasis Level",
    )

    # Save the splines to a pickle file for later use
    pkl.dump(
        {"t": t, "splines": splines, "convolved_splines": convolved_splines},
        open(output_path, "wb"),
    )

    # Display the final plot
    fig.show()


if __name__ == "__main__":
    # --- Main script execution starts here ---

    # Define the path to your data file
    json_file_path = "./raga_data_jog/jog.json"

    # Load the data from the JSON file
    try:
        raga_data = json.load(open(json_file_path))
    except FileNotFoundError:
        print(
            f"Error: The file '{json_file_path}' was not found. Please ensure the path is correct."
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
    spline_output_path = "./model_data_jog/splines.pkl"
    generate_plots_and_save_splines(emphasis_df, *generate_splines(emphasis_df), spline_output_path)

    print(f"\nSpline models saved to '{spline_output_path}'")
