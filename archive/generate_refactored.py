# -*- coding: utf-8 -*-
"""
Refactored script for generating musical phrases based on a transition probability matrix,
emphasis curves derived from splines, and saving the output as text and MIDI.
Includes visualization, configuration flags, and logging. Version focused on conciseness.
"""

import pickle as pkl
import json
import numpy as np
import mido
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import logging
import traceback

# --- Logging Configuration ---
LOG_LEVEL = logging.DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(funcName)s (:%(lineno)d) \n\t %(message)s"
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)  # Global logger

# Add file logging
file_handler = logging.FileHandler("generate_refactored.log")
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(file_handler)

# --- Configuration Flags ---
ENABLE_VISUALIZATION = True
ENABLE_TEMPERATURE_SCALING = True

# --- Constants ---
DEFAULT_TPM_PATH = "tpm.npy"
DEFAULT_AMRIT_PATH = "amrit.json"
DEFAULT_SPLINES_PATH = "splines.pkl"
DEFAULT_SWARS_PATH = "swars.json"
OUTPUT_PHRASES_FILE = "phrases_refactored.txt"
OUTPUT_MIDI_FILE = "amrit_refactored.mid"
NUM_PHRASES = 25
MIDI_BPM = 100
MIDI_TONIC_NOTE = 56
MIDI_TICKS_PER_BEAT = 480
NOTE_SELECTION_TEMPERATURE = 1.3
MIN_NOTE_DURATION_THRESHOLD = 0.6
SPLINE_DOMAIN_MAX = 74  # Domain used for scaling time for spline evaluation

# --- Utility Functions ---


def clip_value(value, min_val, max_val):
    """Clips a value to be within the specified minimum and maximum range."""
    return np.clip(value, min_val, max_val)  # Use numpy clip


def load_data(tpm_path, amrit_path, splines_path, swars_path):
    """Loads necessary data files: TPM, Amrit JSON, Splines, and Swars JSON."""
    logger.info(
        f"Loading data: TPM='{tpm_path}', Amrit='{amrit_path}', Splines='{splines_path}', Swars='{swars_path}'"
    )
    try:
        tpm = np.load(tpm_path)
        with open(amrit_path, "r") as f:
            amrit_data = json.load(f)
        with open(splines_path, "rb") as f:
            splines_data = pkl.load(f)
        with open(swars_path, "r") as f:
            swars_data = json.load(f)

        swars_list = amrit_data.get("notes", []) + ["|"]
        convolved_splines = splines_data.get("convolved_splines")
        swar2midi_map = swars_data.get("swar2midi")

        # Basic validation
        if tpm is None:
            raise ValueError("TPM data failed to load.")
        if not swars_list or swars_list == ["|"]:
            raise ValueError("Swars list is empty or invalid.")
        if convolved_splines is None or not hasattr(convolved_splines, "__iter__"):
            raise ValueError("Splines data failed to load or is not iterable.")
        if swar2midi_map is None:
            raise ValueError("Swar-to-MIDI map failed to load.")

        logger.info(
            f"Loaded {len(swars_list)} swars, {len(convolved_splines)} splines, {len(swar2midi_map)} MIDI mappings."
        )
        logger.debug(f"TPM shape: {tpm.shape}")

        # TPM Shape Validation
        expected_rows = len(swars_list)
        expected_cols_notes = expected_rows - 1
        expected_cols_pipe = expected_rows
        if tpm.shape[0] != expected_rows:
            logger.warning(f"TPM row count {tpm.shape[0]} != expected {expected_rows}.")
        elif tpm.shape[1] not in [expected_cols_notes, expected_cols_pipe]:
            logger.warning(
                f"TPM column count {tpm.shape[1]} not expected ({expected_cols_notes} or {expected_cols_pipe})."
            )
        else:
            logger.debug("TPM shape consistent.")

        return tpm, swars_list, convolved_splines, swar2midi_map

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.exception(f"Error loading/parsing data: {e}")
        raise


def calculate_emphasis(t, t_min, t_max, convolved_splines):
    """Calculates the normalized emphasis distribution for swars at time t."""
    if (
        not convolved_splines
        or not hasattr(convolved_splines, "__iter__")
        or not convolved_splines
    ):
        logger.error("Cannot calculate emphasis: Invalid convolved splines data.")
        raise ValueError("Invalid convolved splines data.")

    # Scale time t to spline domain [0, SPLINE_DOMAIN_MAX]
    t_scaled = clip_value(
        (t - t_min) / (t_max - t_min) * SPLINE_DOMAIN_MAX, 0, SPLINE_DOMAIN_MAX
    )
    logger.debug(f"Calculating emphasis for t={t:.4f}, scaled to {t_scaled:.4f}")

    try:
        # Evaluate splines
        emphasis_vector = np.array([spline(t_scaled) for spline in convolved_splines])
    except Exception as e:
        logger.error(f"Spline evaluation error at t_scaled={t_scaled}: {e}")
        num_notes = len(convolved_splines)
        logger.warning("Using uniform emphasis as fallback.")
        return np.ones(num_notes) / num_notes if num_notes > 0 else np.array([])

    # Clean (NaN, negative) and normalize
    emphasis_vector[np.isnan(emphasis_vector) | (emphasis_vector < 0)] = 0
    total_emphasis = np.sum(emphasis_vector)

    if total_emphasis > 1e-9:
        emphasis_vector /= total_emphasis
    else:
        logger.warning(
            f"Total emphasis near zero at t={t}. Using uniform distribution."
        )
        num_notes = len(convolved_splines)
        emphasis_vector = (
            np.ones(num_notes) / num_notes if num_notes > 0 else np.array([])
        )

    logger.debug(f"Calculated emphasis vector (sum={np.sum(emphasis_vector):.4f})")
    return emphasis_vector


def create_emphasized_tpm(tpm_orig, emphasis, swars_list):
    """Creates a modified TPM based on emphasis, adding a 'Retry' state column."""
    num_swars = len(swars_list)
    num_notes = num_swars - 1

    if len(emphasis) != num_notes:
        logger.error(
            f"Emphasis vector length ({len(emphasis)}) != number of notes ({num_notes})."
        )
        raise ValueError("Emphasis vector length mismatch.")

    has_pipe_col = tpm_orig.shape[1] == num_swars
    new_tpm = np.zeros((num_swars, num_notes + 2))  # Shape (N, N+1) with Retry col

    # 1. Transitions TO '|' (col index -2)
    if has_pipe_col:
        new_tpm[:, -2] = tpm_orig[:, -1]
    # else: column remains zero

    # 2. Transitions FROM '|' (row index -1) to notes (cols 0 to N-2)
    try:
        new_tpm[-1, :-2] = tpm_orig[-1, :num_notes]
    except IndexError as e:
        logger.error(f"Error accessing tpm_orig[-1, :num_notes]: {e}. Check TPM dims.")
        raise

    # 3. Transitions BETWEEN Notes (cols 0 to N-2) scaled by target emphasis
    col_emphasis = np.where(np.isclose(emphasis, 0, atol=5e-3), 0, emphasis)[None, :]
    new_tpm[:, :num_notes] = tpm_orig[:, :num_notes] * col_emphasis

    # 4. Scale transitions TO '|' (col index -2) based on source emphasis (original logic)
    new_tpm[:-1, -2] *= col_emphasis.flatten()

    # --- Row Normalization & Retry Probability (last col) ---
    new_tpm = np.where(
        np.isclose(new_tpm, 0, atol=1e-6), 0, new_tpm
    )  # Clean near-zeros

    for i in range(num_swars):
        row_known = new_tpm[i, :-1]  # Probs for notes + '|'
        row_sum_known = np.sum(row_known)

        if not np.isfinite(row_sum_known):
            logger.warning(
                f"Row {i} ('{swars_list[i]}') sum not finite ({row_sum_known}). Forcing retry."
            )
            new_tpm[i, :-1] = 0
            new_tpm[i, -1] = 1.0
            continue

        retry_prob = 1.0 - row_sum_known

        if retry_prob < -1e-7:  # Sum significantly > 1, normalize
            logger.warning(
                f"Row {i} ('{swars_list[i]}') sum {row_sum_known:.4f} > 1. Renormalizing."
            )
            if row_sum_known > 1e-9:
                new_tpm[i, :-1] /= row_sum_known
            else:
                new_tpm[i, :-1] = 0  # Avoid division by zero if sum was tiny but > 1
            new_tpm[i, -1] = 0
        elif retry_prob < 1e-9:  # Sum is ~1 or retry prob negative but tiny
            new_tpm[i, -1] = 0
        else:  # Valid positive retry probability
            new_tpm[i, -1] = retry_prob

        # Final check and forced normalization if needed
        final_sum = np.sum(new_tpm[i, :])
        if not np.isclose(final_sum, 1.0):
            logger.warning(
                f"Final sum row {i} ('{swars_list[i]}') is {final_sum:.4f} != 1.0. Force normalizing."
            )
            if final_sum > 1e-9:
                new_tpm[i, :] /= final_sum
            else:
                new_tpm[i, :] = 0
                new_tpm[i, -1] = 1.0  # Error case

    logger.debug("Created emphasized TPM.")
    return new_tpm


def generate_single_phrase(
    phrase_num,
    num_total_phrases,
    tpm_orig,
    swars_list,
    convolved_splines,
    enable_temp_scaling=ENABLE_TEMPERATURE_SCALING,
    temp=NOTE_SELECTION_TEMPERATURE,
):
    """Generates a single musical phrase."""
    logger.info(f"Generating phrase {phrase_num}/{num_total_phrases - 1}")
    t_phrase_start = phrase_num
    t_phrase_end = phrase_num + 1
    t_emphasis = clip_value(phrase_num + np.random.normal(0, 0.3), 0, num_total_phrases)

    try:
        emphasis_vector = calculate_emphasis(
            t_emphasis, 0, num_total_phrases, convolved_splines
        )
        emphasized_tpm = create_emphasized_tpm(tpm_orig, emphasis_vector, swars_list)
    except Exception as e:
        logger.error(f"Failed to create emphasis/TPM for phrase {phrase_num}: {e}")
        return None, None  # Cannot proceed

    # --- Starting Note Selection ---
    num_notes = len(swars_list) - 1
    available_start_notes = swars_list[:-1]
    if not available_start_notes:
        logger.error("No notes available to start phrase.")
        return None, None

    start_note_probs = emphasis_vector.copy()
    start_note_probs[np.isnan(start_note_probs) | (start_note_probs < 0)] = 0
    prob_sum = np.sum(start_note_probs)
    if prob_sum < 1e-9:
        logger.warning("Start note probs sum near zero. Using uniform.")
        start_note_probs = (
            np.ones(num_notes) / num_notes if num_notes > 0 else np.array([])
        )
    else:
        start_note_probs /= prob_sum

    current_note = np.random.choice(available_start_notes, p=start_note_probs)

    # Ensure start note can transition out
    start_note_idx = swars_list.index(current_note)
    retry_start_count = 0
    max_start_retries = len(available_start_notes) * 2
    while (
        np.sum(emphasized_tpm[start_note_idx, :-1]) < 1e-9
        and retry_start_count < max_start_retries
    ):
        logger.warning(
            f"Start note '{current_note}' has no outgoing transitions. Reselecting."
        )
        current_note = np.random.choice(available_start_notes, p=start_note_probs)
        start_note_idx = swars_list.index(current_note)
        retry_start_count += 1
    if retry_start_count >= max_start_retries:
        logger.error(
            f"Cannot find valid start note after {max_start_retries} retries. Skipping phrase {phrase_num}."
        )
        viz_data_fail = {
            "emphasis": emphasis_vector,
            "emphasized_tpm": emphasized_tpm,
            "original_tpm": tpm_orig,
        }
        return None, viz_data_fail

    logger.debug(f"Phrase {phrase_num} initial note: {current_note}")

    # --- Phrase Generation Loop ---
    phrase_notes = [(0, "R")]
    previous_note = None
    current_time = t_phrase_start  # Time used for duration calculation
    max_steps = 200
    generated_sequence = [current_note]

    for step_count in range(max_steps):
        current_note_idx = swars_list.index(current_note)
        probs = emphasized_tpm[current_note_idx].copy()

        # Validate & normalize probs
        if np.any(np.isnan(probs)) or np.any(probs < 0):
            logger.error(f"Invalid probs from '{current_note}': {probs}. Breaking.")
            break
        prob_sum = np.sum(probs)
        if prob_sum < 1e-9:
            if current_note == "|" and len(phrase_notes) >= 3:
                logger.debug(f"Terminal state '|' reached.")
            else:
                logger.error(
                    f"Zero prob sum from non-terminal '{current_note}'. Breaking."
                )
            break
        probs /= prob_sum

        # Choose next state
        try:
            next_idx = np.random.choice(len(probs), p=probs)
        except ValueError as e:
            logger.error(
                f"Choice error from '{current_note}' with {probs}: {e}. Breaking."
            )
            break

        # Handle Retry State
        if next_idx == len(swars_list):  # Retry column index
            base_probs = emphasized_tpm[current_note_idx, :-1].copy()
            base_probs[base_probs < 0] = 0
            norm_probs = np.array([])  # Initialize

            if enable_temp_scaling:
                logger.debug(f"Applying temperature scaling for '{current_note}'.")
                with np.errstate(divide="ignore", invalid="ignore"):
                    powered = np.power(base_probs, 1.0 / temp)
                powered[np.isnan(powered) | np.isinf(powered)] = 0
                retry_sum = np.sum(powered)
                if retry_sum > 1e-9:
                    norm_probs = powered / retry_sum
            else:  # No temp scaling
                logger.debug(
                    f"Retrying for '{current_note}' without temperature scaling."
                )
                retry_sum = np.sum(base_probs)
                if retry_sum > 1e-9:
                    norm_probs = base_probs / retry_sum

            # Fallback to uniform if needed
            if norm_probs.size == 0 or np.sum(norm_probs) < 1e-9:
                logger.warning(
                    f"Retry resulted in zero sum for '{current_note}'. Using uniform."
                )
                num_valid = len(base_probs)
                norm_probs = (
                    np.ones(num_valid) / num_valid if num_valid > 0 else np.array([])
                )

            # Update TPM row if possible
            if len(norm_probs) == len(emphasized_tpm[current_note_idx, :-1]):
                emphasized_tpm[current_note_idx, :-1] = norm_probs
                emphasized_tpm[current_note_idx, -1] = 0  # Reset retry prob
            else:
                logger.error(
                    f"Shape mismatch during retry update for '{current_note}'."
                )
            continue  # Re-select note

        # Process Selected Note
        next_note = swars_list[next_idx]

        # End Condition
        if next_note == "|" and len(phrase_notes) >= 3:
            logger.debug(f"Ending phrase {phrase_num} with '|'.")
            break

        # Avoid Repetition
        if len(generated_sequence) > 1 and next_note == generated_sequence[-1]:
            logger.debug(f"Avoiding repetition of '{current_note}'.")
            generated_sequence.pop()
            continue

        generated_sequence.append(next_note)

        # Normal Note Processing (if not '|')
        if next_note != "|":
            time_delta = np.random.normal(0.2, 0.2)
            current_time = clip_value(
                current_time + time_delta, t_phrase_start, t_phrase_end
            )
            try:
                duration_emphasis = calculate_emphasis(
                    current_time, 0, num_total_phrases, convolved_splines
                )
                duration = 10 * duration_emphasis[swars_list.index(next_note)]
            except Exception as e:
                logger.error(
                    f"Error calculating duration for '{next_note}': {e}. Setting duration=0."
                )
                duration = 0

            if duration >= MIN_NOTE_DURATION_THRESHOLD:
                logger.debug(f"Adding note: {next_note} (duration {duration:.2f})")
                phrase_notes.append((duration, next_note))
            else:
                logger.debug(
                    f"Skipping note '{next_note}' (duration {duration:.2f} < {MIN_NOTE_DURATION_THRESHOLD})."
                )

        # Update state
        previous_note = current_note
        current_note = next_note

        # Handle getting stuck on '|' early
        if current_note == "|" and len(phrase_notes) < 3:
            if np.sum(emphasized_tpm[swars_list.index("|"), :-1]) < 1e-9:
                logger.warning(
                    f"Reached '|' early and cannot transition out. Breaking phrase {phrase_num}."
                )
                break
            else:
                logger.debug("Reached '|' early, attempting transition out.")

    else:  # Loop finished without break (max_steps reached)
        logger.warning(
            f"Phrase {phrase_num} reached max steps ({max_steps}). Truncating. Sequence: {' '.join(generated_sequence)}"
        )

    # --- Format Output ---
    viz_data = {
        "emphasis": emphasis_vector,
        "emphasized_tpm": emphasized_tpm,
        "original_tpm": tpm_orig,
    }
    if not phrase_notes:
        logger.warning(
            f"Phrase {phrase_num} generated no valid notes. Sequence: {' '.join(generated_sequence)}"
        )
        return None, viz_data

    phrase_notes = phrase_notes[1:]

    formatted_notes = [f"{duration:.2f}-{note}" for duration, note in phrase_notes]
    final_phrase_string = f"{t_emphasis:.4f} " + " ".join(formatted_notes)
    return final_phrase_string, viz_data


def generate_all_phrases(
    num_phrases_to_gen, tpm_orig, swars_list, convolved_splines, enable_temp_scaling
):
    """Generates a specified number of phrases."""
    logger.info(f"--- Generating {num_phrases_to_gen} Phrases ---")
    logger.info(
        f"Temperature Scaling: {'Enabled' if enable_temp_scaling else 'Disabled'}"
    )
    all_phrases, visualization_tables = [], []

    for i in range(num_phrases_to_gen):
        phrase_str, viz_data = generate_single_phrase(
            i,
            num_phrases_to_gen,
            tpm_orig,
            swars_list,
            convolved_splines,
            enable_temp_scaling,
        )
        if viz_data:
            visualization_tables.append(viz_data)
        if phrase_str:
            all_phrases.append(phrase_str)
            notes_part = (
                phrase_str.split(" ", 1)[1] if " " in phrase_str else "[No Notes]"
            )
            print(f"  Phrase {i+1}: {notes_part}")  # Keep for progress indication

    return all_phrases, visualization_tables


def save_phrases_to_text(phrases_list, filename):
    """Saves the list of generated phrases to a text file."""
    logger.info(f"Saving {len(phrases_list)} phrases to text file: {filename}")
    try:
        with open(filename, "w") as f:
            f.write("\n".join(phrases_list))
        logger.info(f"Successfully saved phrases to {filename}")
    except IOError as e:
        logger.error(f"Error writing phrases to {filename}: {e}")


def create_midi_file(
    phrases_list,
    swar2midi_map,
    tonic,
    bpm,
    filename,
    ticks_per_beat=MIDI_TICKS_PER_BEAT,
):
    """Creates a MIDI file from the generated phrases."""
    if not phrases_list:
        logger.warning("No phrases generated, skipping MIDI.")
        return
    if not swar2midi_map:
        logger.error("Swar->MIDI map missing, skipping MIDI.")
        return

    logger.info(f"Creating MIDI file: {filename}")
    try:
        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        track = mido.MidiTrack()
        mid.tracks.append(track)

        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
        track.append(
            mido.Message("program_change", program=105, channel=0, time=0)
        )  # Sitar

        notes_added = 0
        for phrase_index, phrase_str in enumerate(phrases_list):
            notes_part = phrase_str.split(" ", 1)[1] if " " in phrase_str else ""
            if not notes_part:
                continue

            notes_in_phrase = 0
            for note_info in notes_part.split(" "):
                if not note_info or "-" not in note_info:
                    continue
                try:
                    duration_str, pitch_swar = note_info.split("-")
                    duration_beats = float(duration_str)
                    offset = swar2midi_map.get(pitch_swar)

                    if offset is None or not isinstance(offset, (int, float)):
                        continue
                    midi_note = int(round(offset + tonic))
                    if not (0 <= midi_note <= 127):
                        continue
                    note_ticks = int(round(duration_beats * ticks_per_beat))
                    if note_ticks <= 0:
                        continue

                    # Add note on/off pair with small gap before note_on
                    track.append(
                        mido.Message(
                            "note_on", note=midi_note, velocity=70, channel=0, time=10
                        )
                    )
                    track.append(
                        mido.Message(
                            "note_off",
                            note=midi_note,
                            velocity=64,
                            channel=0,
                            time=note_ticks,
                        )
                    )
                    notes_added += 1
                    notes_in_phrase += 1
                except Exception as e:
                    logger.warning(f"Error processing MIDI note '{note_info}': {e}")
                    continue

            # Add inter-phrase pause if notes were added
            if notes_in_phrase > 0:
                pause_ticks = int(round(1.0 * ticks_per_beat))  # 1 beat pause
                track.append(
                    mido.Message(
                        "control_change",
                        control=123,
                        value=0,
                        channel=0,
                        time=pause_ticks,
                    )
                )  # All notes off CC

        track.append(mido.MetaMessage("end_of_track", time=0))
        mid.save(filename)
        logger.info(
            f"Successfully saved MIDI file with {notes_added} notes to {filename}"
        )

    except ImportError:
        logger.critical("'mido' library not found. Cannot create MIDI.")
    except Exception as e:
        logger.exception(f"Error creating MIDI file {filename}: {e}")


def create_visualization(viz_tables, swars_list, num_phrases_generated):
    """
    Creates an animated Plotly visualization of TPMs and Emphasis,
    using a specific visual style for the figures.
    """

    if not viz_tables:
        logger.warning("No visualization data available to create a plot.")
        return

    logger.info("Creating Plotly visualization...")

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
        tpm_labels = swars_list + ["Retry"]

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
            coloraxis1=dict(
                colorscale="hot", cmin=0, cmax=1
            ),
            coloraxis2=dict(
                colorscale="hot", cmin=0, cmax=1
            ),
            coloraxis3=dict(
                colorscale="hot",
                cmin=0,
                cmax=1,
            ),
        )

        figs.frames = frames

        # Directory to save frames
        frames_dir = "frames"

        # Create directory if it doesn't exist
        os.makedirs(frames_dir, exist_ok=True)

        # Save each frame as a separate PNG image
        for i, frame in enumerate(frames):
            figs.update(frames=[frame])
            image_filename = os.path.join(frames_dir, f"frame_{i}.png")
            figs.write_image(image_filename, format="png")
            logger.info(f"Saved frame {i} to {image_filename}")

        figs.frames = frames
        figs.show()
        logger.info("Visualization successfully shown.")

    except Exception as e:
        logger.exception(f"Unexpected error during visualization: {e}")


# --- Main Execution ---


def main():
    """Main function to orchestrate the phrase generation and output."""
    logger.info("Starting music generation process...")
    logger.info(f"Visualization Enabled: {ENABLE_VISUALIZATION}")
    logger.info(f"Temperature Scaling Enabled: {ENABLE_TEMPERATURE_SCALING}")

    # Define file paths
    paths = {
        p + "_path": globals()[f"DEFAULT_{p.upper()}_PATH"]
        for p in ["tpm", "amrit", "splines", "swars"]
    }

    # Check files exist
    missing = [p for p, f in paths.items() if not os.path.exists(f)]
    if missing:
        logger.critical(f"Missing input file(s): {', '.join(missing)}. Exiting.")
        return

    # Load data
    try:
        tpm_orig, swars_list, convolved_splines, swar2midi_map = load_data(**paths)
        logger.info("Data loaded successfully.")
    except Exception as e:
        logger.critical(
            "Failed to load/validate data. Exiting."
        )  # Error logged in load_data
        logger.critical(traceback.format_exc())
        logger.critical(e)
        return

    # Generate phrases
    phrases, viz_tables = generate_all_phrases(
        num_phrases_to_gen=NUM_PHRASES,
        tpm_orig=tpm_orig,
        swars_list=swars_list,
        convolved_splines=convolved_splines,
        enable_temp_scaling=ENABLE_TEMPERATURE_SCALING,
    )

    # Save phrases & MIDI
    if phrases:
        save_phrases_to_text(phrases, OUTPUT_PHRASES_FILE)
        if swar2midi_map:
            create_midi_file(
                phrases,
                swar2midi_map,
                MIDI_TONIC_NOTE,
                MIDI_BPM,
                OUTPUT_MIDI_FILE,
                MIDI_TICKS_PER_BEAT,
            )
        else:
            logger.error("Cannot create MIDI: Swar->MIDI map missing.")
    else:
        logger.warning("No phrases generated to save or create MIDI.")

    # Create visualization
    if ENABLE_VISUALIZATION:
        if viz_tables:
            create_visualization(viz_tables, swars_list, NUM_PHRASES)
        else:
            logger.warning("Skipping visualization: No data available.")
    else:
        logger.info("Skipping visualization as per configuration.")

    logger.info("Music generation process finished.")


if __name__ == "__main__":

    main()
