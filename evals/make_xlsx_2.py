import re
import pandas as pd
import numpy as np


def parse_metrics(model_section):
    """Parses the main metrics from a model's log section."""
    metrics = {}
    patterns = {
        "unique_tags": r"% of unique tags ([\d.]+)",
        "avg_phrase_length": r"Average phrase length ([\d.]+)",
        "num_errors": r"Num errors ([\d.]+)",
        "avg_vocab_size": r"Average vocab size ([\d.]+)",
        "std_phrase_length": r"Std phrase length ([\d.]+)",
        "std_vocab_size": r"Std vocab size ([\d.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, model_section)
        metrics[key] = float(match.group(1)) if match else np.nan
    return metrics


def parse_arrays(model_section):
    """Parses the structural fidelity numpy arrays from a model's log section."""
    # Find all numpy arrays in the section. They might span multiple lines.
    array_strings = re.findall(r"array\(\[.*?\]\)", model_section, re.DOTALL)
    if len(array_strings) < 2:
        return None, None

    # The first array is the mean distance, the second is the std dev.
    # Clean up the string and convert to a numpy array of floats.
    mean_dist_str = (
        array_strings[0].replace("array([", "").replace("])", "").replace("\n", "")
    )
    std_dist_str = (
        array_strings[1].replace("array([", "").replace("])", "").replace("\n", "")
    )

    mean_dist = np.fromstring(mean_dist_str, sep=",")
    std_dist = np.fromstring(std_dist_str, sep=",")

    return mean_dist, std_dist


def parse_vocab_p_values(log_content):
    """Parses the vocabulary t-test p-values."""
    p_values = {}
    pattern = (
        r"--- T-Tests of Vocab Size ---\n(.*?)\n--- T-Tests of Structural Fidelity ---"
    )
    match = re.search(pattern, log_content, re.DOTALL)
    if not match:
        return {}

    section = match.group(1)
    lines = section.strip().split("\n")
    for line in lines:
        parts = line.split(" p-value: ")
        if len(parts) == 2:
            p_values[parts[0].strip()] = float(parts[1])
    return p_values


def parse_fidelity_p_values(log_content):
    """Parses the structural fidelity t-test p-values into a DataFrame."""
    p_values_data = []
    pattern = r"note (\d+) (.*?) vs (.*?): ([\d.e-]+|nan)"
    matches = re.findall(pattern, log_content)

    data = {}
    for note, model1, model2, p_val in matches:
        note = int(note)
        comparison = f"{model1.strip()} vs {model2.strip()}"
        if comparison not in data:
            data[comparison] = {}
        # Handle 'nan' values
        try:
            data[comparison][note] = float(p_val)
        except ValueError:
            data[comparison][note] = np.nan

    df = pd.DataFrame(data)
    df.index.name = "Note"
    return df


def highlight_min_row(s):
    """Highlight the minimum in a row."""
    is_min = s == s.min()
    return ["background-color: #d4edda" if v else "" for v in is_min]  # light green


def highlight_p_values(val):
    """Highlight p-values based on significance."""
    if pd.isna(val):
        return ""
    if val < 0.001:
        return "background-color: #f8d7da"  # light red for highly significant
    elif val < 0.0083:
        return "background-color: #fff3cd"  # light yellow for significant
    else:
        return "background-color: #e2e3e5"  # light gray for not significant


def main(log_file_path, output_excel_path):
    """Main function to parse log and generate Excel report."""
    with open(log_file_path, "r") as f:
        content = f.read()

    # Split the log file by model runs
    sections = re.split(r"Running with ", content)
    model_data = {}
    model_names_map = {
        "emphasis, tags, and hybrid enabled": "Full Model",
        "emphasis and tags enabled": "Tags-Only",
        "emphasis enabled": "Emphasis-Only",
        "base Markov model": "Base Model",
    }

    # Parse data for each model
    for section in sections[1:]:  # Skip the first split part (before any model)
        for key, name in model_names_map.items():
            if section.startswith(key):
                model_data[name] = {
                    "metrics": parse_metrics(section),
                    "struct_fid_mean": parse_arrays(section)[0],
                    "struct_fid_std": parse_arrays(section)[1],
                }
                break

    # --- Create Excel Writer ---
    writer = pd.ExcelWriter(output_excel_path, engine="xlsxwriter")

    # --- Sheet 1: Metrics ---
    # Create metrics DataFrames
    mean_metrics_data = {
        name: {
            "Unique Tags (%)": data["metrics"]["unique_tags"],
            "Vocab Size": data["metrics"]["avg_vocab_size"],
            "Phrase Length": data["metrics"]["avg_phrase_length"],
            "Rules Broken": data["metrics"]["num_errors"],
        }
        for name, data in model_data.items()
    }
    std_metrics_data = {
        name: {
            "Vocab Size": data["metrics"]["std_vocab_size"],
            "Phrase Length": data["metrics"]["std_phrase_length"],
        }
        for name, data in model_data.items()
    }

    df_mean_metrics = pd.DataFrame(mean_metrics_data).T
    df_std_metrics = pd.DataFrame(std_metrics_data).T

    # Create Vocab p-values DataFrame
    vocab_p_values = parse_vocab_p_values(content)
    df_vocab_p = pd.DataFrame(
        list(vocab_p_values.items()), columns=["Comparison", "P-Value"]
    )

    # Write metrics to sheet 1
    df_mean_metrics.to_excel(writer, sheet_name="Metrics", startrow=1, startcol=0)
    df_std_metrics.to_excel(writer, sheet_name="Metrics", startrow=8, startcol=0)
    df_vocab_p.to_excel(
        writer,
        sheet_name="Metrics",
        startrow=1,
        startcol=len(df_mean_metrics.columns) + 2,
        index=False,
    )

    # Add titles to Sheet 1
    worksheet1 = writer.sheets["Metrics"]
    worksheet1.write("A1", "Mean Metrics")
    worksheet1.write("A8", "Standard Deviation of Metrics")
    worksheet1.write(
        f'{chr(ord("A") + len(df_mean_metrics.columns) + 2)}1',
        "Vocab Size T-Test P-Values",
    )

    # --- Sheet 2: Structural Fidelity ---
    # Create raw distance DataFrame
    struct_fid_mean_data = {
        name: data["struct_fid_mean"] for name, data in model_data.items()
    }
    df_struct_fid_mean = pd.DataFrame(
        struct_fid_mean_data,
        index=[
            f"Note {i}" for i in range(len(next(iter(struct_fid_mean_data.values()))))
        ],
    )

    # Create std dev DataFrame
    struct_fid_std_data = {
        name: data["struct_fid_std"] for name, data in model_data.items()
    }
    df_struct_fid_std = pd.DataFrame(
        struct_fid_std_data,
        index=[
            f"Note {i}" for i in range(len(next(iter(struct_fid_std_data.values()))))
        ],
    )

    # Create fidelity p-values DataFrame
    df_fidelity_p = parse_fidelity_p_values(content)

    # Apply styling and write to sheet 2
    df_struct_fid_mean.style.apply(highlight_min_row, axis=1).to_excel(
        writer, sheet_name="Structural Fidelity", startrow=1, startcol=0
    )
    df_struct_fid_std.to_excel(
        writer,
        sheet_name="Structural Fidelity",
        startrow=df_struct_fid_mean.shape[0] + 5,
        startcol=0,
    )
    if not df_fidelity_p.empty:
        df_fidelity_p.style.map(highlight_p_values).to_excel(
            writer,
            sheet_name="Structural Fidelity",
            startrow=1,
            startcol=df_struct_fid_mean.shape[1] + 2,
        )

    # Add titles to Sheet 2
    worksheet2 = writer.sheets["Structural Fidelity"]
    worksheet2.write("A1", "Raw Distance Values (Lower is Better)")
    worksheet2.write(
        f"A{df_struct_fid_mean.shape[0] + 5}", "Standard Deviations of Distance"
    )
    if not df_fidelity_p.empty:
        worksheet2.write(
            f'{chr(ord("A") + df_struct_fid_mean.shape[1] + 2)}1',
            "Structural Fidelity T-Test P-Values",
        )

    # --- Save the Excel file ---
    writer.close()
    print(f"Successfully created Excel file at: {output_excel_path}")


if __name__ == "__main__":
    # Use the uploaded log file and define an output path
    log_file = "1019_eval_log_jog.txt"
    output_excel = "1019_evals_jog.xlsx"
    main(log_file, output_excel)
