# -*- coding: utf-8 -*-
"""
Refactored script for generating musical phrases based on a transition probability matrix,
emphasis curves derived from splines, and saving the output as text and MIDI.
Includes visualization, configuration flags, and logging. Version focused on conciseness.
"""

import json
import logging
import os
import pickle as pkl
import sys
from collections import defaultdict

import mido
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Logging Configuration ---
LOG_LEVEL = logging.ERROR  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s.%(msecs)03d - %(levelname)s - %(funcName)s (:%(lineno)d) \n\t %(message)s"
logger = logging.getLogger(__name__)  # Global logger
logger.setLevel(logging.DEBUG)
logger.handlers = []

# Add file logging
file_handler = logging.FileHandler("./output/debug.log", mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))
logger.addHandler(file_handler)

# StreamHandler — logs only errors and worse
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(LOG_LEVEL)
stdout_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))
logger.addHandler(stdout_handler)

name = sys.argv[-1] if len(sys.argv) > 1 else "amrit"

logger.error("Generating music for " + name)

# --- Constants ---
DEFAULT_TPM_PATH = f"./model_data_{name}/tpm.npy"
DEFAULT_RAGADATA_PATH = f"./raga_data_{name}/{name}.json"
DEFAULT_TAG_TPM_PATH = f"./model_data_{name}/tpm_tags.npy"
DEFAULT_TAG_PICKLE_PATH = f"./model_data_{name}/tag_tpms.pkl"
DEFAULT_SPLINES_PATH = f"./model_data_{name}/splines.pkl"
DEFAULT_SWARS_PATH = f"./raga_data_{name}/swars.json"
OUTPUT_PHRASES_FILE = f"./output/phrases_{name}.txt"
OUTPUT_MIDI_FILE = f"./output/out_{name}.mid"

MIDI_BPM = 100
MIDI_TONIC_NOTE = 56  # G#3
MIDI_TICKS_PER_BEAT = 480

NOTE_SELECTION_TEMPERATURE = 1.3
MIN_NOTE_DURATION_THRESHOLD = 0.5

SPLINE_DOMAIN_MAX = (
    74 if name == "amrit" else 59
)  # Domain used for scaling time for spline evaluation

# Ablation parameters
ENABLE_EMPHASIS = True  # Enable emphasis calculation
ENABLE_TAGS = True  # Enable tag-based FSM generation
ENABLE_HYBRID = True  # Enable hybrid TPM with tags and emphasis

# --- Configuration Flags ---
ENABLE_VISUALIZATION = True
ENABLE_TEMPERATURE_SCALING = ENABLE_EMPHASIS
ENABLE_TRANSITION_DECAY = False
# --- Data Loading ---

full_tpm = np.load(DEFAULT_TPM_PATH)
raga_data = json.load(open(DEFAULT_RAGADATA_PATH))
splines_data = pkl.load(open(DEFAULT_SPLINES_PATH, "rb"))
swars_data = json.load(open(DEFAULT_SWARS_PATH))

swars_list = raga_data.get("notes", []) + ["|"]
convolved_splines = splines_data.get("convolved_splines")
swar2midi_map = swars_data.get("swar2midi")

# --- Tag stuff ---
tags = raga_data["tags"]
note_tags = raga_data["new_tags"]

tag_tpm = np.load(DEFAULT_TAG_TPM_PATH)

tag_tpm_dict, unique_note_tags, tags_to_time = pkl.load(
    open(DEFAULT_TAG_PICKLE_PATH, "rb")
).values()

# --- Utility Functions ---


def changeRaga(n):

    global name
    global DEFAULT_TPM_PATH
    global DEFAULT_RAGADATA_PATH
    global DEFAULT_TAG_TPM_PATH
    global DEFAULT_TAG_PICKLE_PATH
    global DEFAULT_SPLINES_PATH
    global DEFAULT_SWARS_PATH
    global OUTPUT_PHRASES_FILE
    global OUTPUT_MIDI_FILE
    global SPLINE_DOMAIN_MAX

    name = n
    DEFAULT_TPM_PATH = f"./model_data_{n}/tpm.npy"
    DEFAULT_RAGADATA_PATH = f"./raga_data_{n}/{n}.json"
    DEFAULT_TAG_TPM_PATH = f"./model_data_{n}/tpm_tags.npy"
    DEFAULT_TAG_PICKLE_PATH = f"./model_data_{n}/tag_tpms.pkl"
    DEFAULT_SPLINES_PATH = f"./model_data_{n}/splines.pkl"
    DEFAULT_SWARS_PATH = f"./raga_data_{n}/swars.json"
    OUTPUT_PHRASES_FILE = f"./output/phrases_{n}.txt"
    OUTPUT_MIDI_FILE = f"./output/out_{n}.mid"
    SPLINE_DOMAIN_MAX = (
        74 if name == "amrit" else 59
    )  # Domain used for scaling time for spline evaluation


def swar2midi(swar):
    return swar2midi_map[swar]


def clip_value(value, min_val, max_val):
    """Clips a value to be within the specified minimum and maximum range."""
    return np.clip(value, min_val, max_val)  # Use numpy clip


def calculate_emphasis(t, convolved_splines):
    """Calculates the normalized emphasis distribution for swars at time t."""

    # Scale time t to spline domain [0, SPLINE_DOMAIN_MAX]
    t_scaled = clip_value(t, 0, SPLINE_DOMAIN_MAX)
    logger.debug(f"Calculating emphasis for t={t:.4f}, clipped to {t_scaled:.4f}")

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


def decay_tpm(tpm, current, prev, decay_factor=0.9):
    current = swars_list.index(current)
    prev = swars_list.index(prev)
    tpm[current, prev] *= decay_factor
    tpm = normalize(tpm)
    return tpm


def normalize(tpm):
    for row in tpm:
        if np.sum(row) > 1e-10:
            row /= np.sum(row)
    return tpm


# --- Generating functions ---


def generate_single_phrase(
    timestamp,
    num_total_phrases,
    tpm_orig,
    swars_list,
    convolved_splines,
    enable_temp_scaling=ENABLE_TEMPERATURE_SCALING,
    fsm_tag="I S",
    temp=NOTE_SELECTION_TEMPERATURE,
):
    """Generates a single musical phrase."""
    logger.info(f"Generating phrase {timestamp}/{num_total_phrases - 1}, tag {fsm_tag}")
    t_phrase_start = timestamp
    t_phrase_end = timestamp + 1
    t_emphasis = clip_value(
        timestamp + np.random.normal(0, 0.3), t_phrase_start, t_phrase_end
    )

    try:
        if ENABLE_HYBRID and ENABLE_EMPHASIS:
            emphasis_vector = calculate_emphasis(t_emphasis, convolved_splines)
            mod_tpm = 0.9 * tpm_orig + 0.1 * full_tpm
            emphasized_tpm = create_emphasized_tpm(
                mod_tpm, emphasis_vector, swars_list
            )
        elif ENABLE_EMPHASIS:
            emphasis_vector = calculate_emphasis(t_emphasis, convolved_splines)
            emphasized_tpm = create_emphasized_tpm(
                tpm_orig, emphasis_vector, swars_list
            )

        else:
            emphasis_vector = np.ones(len(swars_list) - 1) / (len(swars_list) - 1)
            emphasized_tpm = create_emphasized_tpm(
                tpm_orig, np.ones(len(swars_list) - 1), swars_list
            )
    except Exception as e:
        logger.error(f"Failed to create emphasis/TPM for phrase {timestamp}: {e}")
        return None, None  # Cannot proceed

    # --- Starting Note Selection ---
    num_notes = len(swars_list) - 1
    available_start_notes = swars_list[:-1]
    if not available_start_notes:
        logger.error("No notes available to start phrase.")
        return None, None

    start_note_probs = emphasized_tpm[-1][:-2].copy()
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
            f"Cannot find valid start note after {max_start_retries} retries. Skipping phrase {timestamp}."
        )
        viz_data_fail = {
            "emphasis": emphasis_vector,
            "emphasized_tpm": emphasized_tpm,
            "original_tpm": tpm_orig,
        }
        return None, viz_data_fail

    logger.debug(f"Phrase {timestamp} initial note: {current_note}")

    current_time = t_phrase_start  # Time used for duration calculation

    time_delta = np.random.normal(0.2, 0.2)
    current_time = clip_value(current_time + time_delta, t_phrase_start, t_phrase_end)
    try:
        if ENABLE_EMPHASIS:
            duration_emphasis = calculate_emphasis(current_time, convolved_splines)
            duration = 10 * duration_emphasis[swars_list.index(current_note)]
        else:
            duration = 1
    except Exception as e:
        logger.error(
            f"Error calculating duration for '{current_note}': {e}. Setting duration=0."
        )
        duration = 0

    if duration >= MIN_NOTE_DURATION_THRESHOLD:
        logger.debug(f"Adding note: {current_note} (duration {duration:.2f})")
    else:
        duration = MIN_NOTE_DURATION_THRESHOLD
        logger.debug(
            f"Clamping note '{current_note}' (duration {duration:.2f} < {MIN_NOTE_DURATION_THRESHOLD})."
        )

    # --- Phrase Generation Loop ---
    phrase_notes = [(duration, current_note)]
    max_steps = 50
    generated_sequence = [current_note]

    for _ in range(max_steps):
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
            tpm_temp_scale(
                enable_temp_scaling,
                temp,
                emphasized_tpm,
                current_note,
                current_note_idx,
            )
            continue  # Re-select note

        # Process Selected Note
        next_note = swars_list[next_idx]

        # End Condition
        if next_note == "|" and len(phrase_notes) >= 3:
            logger.debug(f"Ending phrase {timestamp} with '|'.")
            break

        # Avoid Repetition
        if len(generated_sequence) > 1 and next_note == current_note:
            logger.debug(f"Avoiding repetition of '{current_note}'.")
            tpm_temp_scale(
                enable_temp_scaling,
                temp,
                emphasized_tpm,
                current_note,
                current_note_idx,
            )
            continue

        # Normal Note Processing (if not '|')
        if next_note != "|":
            time_delta = np.random.normal(0.2, 0.2)
            current_time = clip_value(
                current_time + time_delta, t_phrase_start, t_phrase_end
            )
            try:
                if ENABLE_EMPHASIS:
                    duration_emphasis = calculate_emphasis(
                        current_time, convolved_splines
                    )
                    duration = 10 * duration_emphasis[swars_list.index(next_note)]
                else:
                    duration = 1
            except Exception as e:
                logger.error(
                    f"Error calculating duration for '{next_note}': {e}. Setting duration=0."
                )
                duration = 0

            tpm_temp_scale(
                enable_temp_scaling,
                1.1,
                emphasized_tpm,
                next_note,
                next_idx,
            )

            if duration >= MIN_NOTE_DURATION_THRESHOLD:
                logger.debug(f"Adding note: {next_note} (duration {duration:.2f})")
            else:
                logger.debug(
                    f"Adding note '{next_note}' (duration {duration:.2f} < {MIN_NOTE_DURATION_THRESHOLD}). Clamping to {MIN_NOTE_DURATION_THRESHOLD}."
                )
                duration = MIN_NOTE_DURATION_THRESHOLD
            phrase_notes.append((duration, next_note))

        # Handle getting stuck on '|' early
        if next_note == "|" and len(phrase_notes) < 3:
            logger.debug("Reached '|' early, attempting transition out.")

            if emphasized_tpm[current_note_idx, -2] > 1 - 1e-9:
                logger.warning(
                    f"Stuck on '|' with high retry prob from '{current_note}'. Skipping retry."
                )
                break

            tpm_temp_scale(
                enable_temp_scaling,
                temp,
                emphasized_tpm,
                current_note,
                current_note_idx,
            )
            continue

        # Update state
        prev_note = current_note
        current_note = next_note

        logger.info(f"Current sequence: {phrase_notes}")

        if ENABLE_EMPHASIS and ENABLE_HYBRID:
            emphasis_vector += np.ones(len(emphasis_vector)) * 0.001
            emphasis_vector /= np.sum(emphasis_vector)
            emphasis_vector **= 0.95
            emphasis_vector /= np.sum(emphasis_vector)

            # mod_tpm[:, -1] *= 1.02
            emphasized_tpm = create_emphasized_tpm(mod_tpm, emphasis_vector, swars_list)

        if ENABLE_TRANSITION_DECAY:
            emphasized_tpm = decay_tpm(emphasized_tpm, current_note, prev_note)

    else:  # Loop finished without break (max_steps reached)
        logger.warning(
            f"Phrase {timestamp} reached max steps ({max_steps}). Truncating. Sequence: {phrase_notes}"
        )

    # --- Format Output ---
    viz_data = {
        "emphasis": emphasis_vector,
        "emphasized_tpm": emphasized_tpm,
        "original_tpm": tpm_orig,
    }
    if not phrase_notes:
        logger.warning(f"Phrase {timestamp} generated no valid notes.")
        return None, viz_data

    formatted_notes = [f"{duration:.2f}-{note}" for duration, note in phrase_notes]
    final_phrase_string = f"{fsm_tag} {t_emphasis:.4f} " + " ".join(formatted_notes)

    logger.info(f"Generated final phrase: {final_phrase_string}")
    return final_phrase_string, viz_data


def tpm_temp_scale(
    enable_temp_scaling, temp, emphasized_tpm, current_note, current_note_idx
):
    base_probs = emphasized_tpm[current_note_idx, :-1].copy()
    base_probs[base_probs <= 0] = 0
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
        logger.debug(f"Retrying for '{current_note}' without temperature scaling.")
        retry_sum = np.sum(base_probs)
        if retry_sum > 1e-9:
            norm_probs = base_probs / retry_sum

        # Fallback to uniform if needed
    if norm_probs.size == 0 or np.sum(norm_probs) < 1e-9:
        logger.warning(
            f"Retry resulted in zero sum for '{current_note}'. Using uniform."
        )
        num_valid = len(base_probs)
        norm_probs = np.ones(num_valid) / num_valid if num_valid > 0 else np.array([])

        # Update TPM row if possible
    if len(norm_probs) == len(emphasized_tpm[current_note_idx, :-1]):
        emphasized_tpm[current_note_idx, :-1] = norm_probs
        emphasized_tpm[current_note_idx, -1] = 0  # Reset retry prob
    else:
        logger.error(f"Shape mismatch during retry update for '{current_note}'.")

    logger.info(f"Retry probabilities for '{current_note}': {norm_probs}")


def create_tags():
    initial_state = "I S"
    gen = [initial_state]

    i = 0
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

    return gen


def generate_all_phrases(
    swars_list,
    convolved_splines,
    enable_temp_scaling=ENABLE_TEMPERATURE_SCALING,
    gen=[],
):
    """Generates a specified number of phrases."""
    logger.info("--- Generating Phrases ---")
    logger.info("Generating Tag Sequence using FSM...")

    all_phrases, visualization_tables = [], []
    times = []
    if not gen:
        time_to_tag = {v: k for k, v in tags_to_time.items()}

        if name == "amrit":
            l = 43
        elif name == "jog":
            l = 47

        times = np.linspace(
            0, SPLINE_DOMAIN_MAX, l
        )  # Tag-based vistaars are ~43 (median) phrases long
        gen = []
        for t in times:
            snap = min(
                time_to_tag, key=lambda x: abs(x - t) if x <= t else float("inf")
            )
            gen.append(time_to_tag[snap])
    logger.info(f"--- Generating {len(gen)} Phrases ---")
    logger.info(
        f"Temperature Scaling: {'Enabled' if enable_temp_scaling else 'Disabled'}"
    )

    timestamp = 0
    for k, state in enumerate(gen):
        if ENABLE_TAGS:
            tpm_orig = tag_tpm_dict[state]

            if state[0] in ["I", "F", "R", "C"]:
                timestamp = tags_to_time[state]
            else:
                timestamp += 1

        else:
            tpm_orig = full_tpm
            timestamp = times[k]

        phrase_str, viz_data = generate_single_phrase(
            timestamp,
            len(gen),
            tpm_orig,
            swars_list,
            convolved_splines,
            enable_temp_scaling,
            state,
        )
        if viz_data:
            visualization_tables.append(viz_data)
        if phrase_str:
            all_phrases.append(phrase_str)
            notes_part = (
                phrase_str.split(" ", 1)[1] if " " in phrase_str else "[No Notes]"
            )
            # print(f"  Phrase {i+1}: {notes_part}")  # Keep for progress indication
    return all_phrases, visualization_tables


# --- Evaluation functions ---


def delta(a, b):
    return 1 if a == b else 0


def evaluate_phrase(phrase_str, swars_list, convolved_splines):
    """Evaluates the quality of a generated phrase."""

    total = 0
    evals = defaultdict(bool)
    tag, emphasis_note = phrase_str.split(" ")[:2]
    phrase_str = phrase_str.split(" ")[3:]
    notes = [s.split("-")[1] for s in phrase_str]

    # Make sure phrase isn't too short
    if len(phrase_str) < 3:
        evals["err_length"] = True

    # Intro phrases MUST contain the note they introduce!
    if tag == "I" and emphasis_note not in notes:
        evals["err_intro"] = True

    # Phrases in a vistaar should not go far above the note they are focusing on
    max_note = max(swars_list.index(note) for note in notes)
    emp_note = swars_list.index(emphasis_note)
    if (
        (max_note > emp_note and tag == "I")
        or (max_note > emp_note + 1 and tag == "E")
        or (max_note > emp_note + 2 and tag == "T")
    ):
        evals["err_gap"] = True

    # Autocorrelation (lag 1, 2, 3)
    for lag in range(1, 4):
        acr = 0
        for i in range(len(notes) - lag):
            acr += delta(notes[i], notes[i + lag])

        acr /= len(notes)
        evals[f"acr_lag_{lag}"] += acr

    return tag, emphasis_note, notes, evals


def evaluate_all_phrases(phrases_list, swars_list, convolved_splines):
    """Evaluates the quality of all generated phrases."""

    evals = {"phrase_evals": []}
    tags = []
    emphasis_notes = []
    notes_lists = []
    for phrase_str in phrases_list:
        tag, emphasis_note, notes, phrase_evals = evaluate_phrase(
            phrase_str, swars_list, convolved_splines
        )
        phrase_evals["tag"] = tag
        phrase_evals["emphasis_note"] = emphasis_note
        evals["phrase_evals"].append(phrase_evals)
        tags.append(tag)
        emphasis_notes.append(emphasis_note)
        notes_lists.append(notes)

    all_notes = [note for notes in notes_lists for note in notes]
    # Autocorrelation (lag 1, 2, 3)
    for lag in range(1, 4):
        acr = 0
        for i in range(len(all_notes) - lag):
            acr += delta(all_notes[i], all_notes[i + lag])

        acr /= len(all_notes)
        evals.setdefault(f"full_acr_lag_{lag}", 0)
        evals[f"full_acr_lag_{lag}"] += acr

    if notes_lists[-1][-1] != "S":
        evals["err_last_note"] = True

    num_unique_tags = len(
        set([tags[i] + " " + emphasis_notes[i] for i in range(len(tags))])
    )
    evals["unique_tags_%"] = num_unique_tags / len(unique_note_tags) * 100
    evals["num_errors"] = [x for d in evals["phrase_evals"] for x in d.values()].count(
        True
    )
    return evals, 0


# --- Save/Plot functions ---


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


def create_visualization(viz_tables, swars_list):
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


# --- Main Execution ---


def main():
    """Main function to orchestrate the phrase generation and output."""
    logger.info("Starting music generation process...")
    logger.info(f"Visualization Enabled: {ENABLE_VISUALIZATION}")
    logger.info(f"Temperature Scaling Enabled: {ENABLE_TEMPERATURE_SCALING}")

    # Define file paths
    paths = [
        DEFAULT_TPM_PATH,
        DEFAULT_RAGADATA_PATH,
        DEFAULT_SPLINES_PATH,
        DEFAULT_SWARS_PATH,
    ]

    # Check files exist
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        logger.critical(f"Missing input file(s): {', '.join(missing)}. Exiting.")
        return

    # Generate phrases
    phrases, viz_tables = generate_all_phrases(
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
            create_visualization(viz_tables, swars_list)
        else:
            logger.warning("Skipping visualization: No data available.")
    else:
        logger.info("Skipping visualization as per configuration.")

    logger.info("Music generation process finished.")


def main_with_eval():
    """Main function to orchestrate the phrase generation and output."""
    logger.info("Starting music generation process with evaluation...")
    logger.info(f"Visualization Enabled: {ENABLE_VISUALIZATION}")
    logger.info(f"Temperature Scaling Enabled: {ENABLE_TEMPERATURE_SCALING}")

    # TODO
    total_iter = 0
    score = -1e10

    if ENABLE_TAGS:
        gen = create_tags()
    else:
        gen = []

    best_phrases = []
    best_table = []
    best_eval = -1e10
    best_idx = -1

    while score < -3 and total_iter < 1:
        logger.info("Eval Iteration: " + str(total_iter))
        # Generate phrases
        phrases, viz_tables = generate_all_phrases(
            swars_list=swars_list,
            convolved_splines=convolved_splines,
            enable_temp_scaling=ENABLE_TEMPERATURE_SCALING,
            gen=gen,
        )

        current_eval, score = evaluate_all_phrases(
            phrases, swars_list, convolved_splines
        )
        logger.error(
            f"Evaluation for iteration {total_iter}: {json.dumps(current_eval, indent=2)}"
        )

        if score > best_eval:
            best_eval = score
            best_phrases = phrases
            best_table = viz_tables
            best_idx = total_iter

        total_iter += 1
    # Save phrases & MIDI
    if best_phrases:
        logger.error(f"Saving best phrases (iter {best_idx})...")
        save_phrases_to_text(best_phrases, OUTPUT_PHRASES_FILE)
        if swar2midi_map:
            create_midi_file(
                best_phrases,
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
        if best_table:
            create_visualization(best_table, swars_list)
        else:
            logger.warning("Skipping visualization: No data available.")
    else:
        logger.info("Skipping visualization as per configuration.")

    logger.info("Music generation process finished.")


if __name__ == "__main__":

    main_with_eval()
