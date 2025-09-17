import pandas as pd
import re
import warnings

# Suppress a common UserWarning from openpyxl regarding styles, which is not critical here.
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def parse_log_file(log_content):
    """
    Parses the entire log file content and organizes data into a dictionary,
    with each key representing a separate evaluation and its corresponding DataFrames.
    """
    print("Starting to parse log file...")
    all_data = {}

    # Define model names for consistent mapping across all sections.
    model_name_map = {
        "emphasis, tags, and hybrid enabled": "Full",
        "emphasis and tags enabled": "Tags-Only",
        "emphasis enabled": "Emphasis-Only",
        "base Markov model": "Base",
        "ground truth vistaar": "Ground Truth",
    }

    # --- Section 1: Phrase Rules ---
    try:
        section_content = re.search(
            r"--- Ablative Evaluation I - Phrase Rules ---(.*?)--- Ablative Evaluation II",
            log_content,
            re.DOTALL,
        ).group(1)
        metrics_list = []
        for desc, name in model_name_map.items():
            pattern = (
                re.escape(desc)
                + r".*?Average ACR Lag 1:\s+([\d.e-]+).*?Average ACR Lag 2:\s+([\d.e-]+).*?Average ACR Lag 3:\s+([\d.e-]+).*?Total rules broken:\s+(\d+).*?Rules broken per vistaar:\s+([\d.e-]+)"
            )
            match = re.search(pattern, section_content, re.DOTALL)
            if match:
                metrics_list.append(
                    {
                        "Model": name,
                        "Avg ACR Lag 1": float(match.group(1)),
                        "Avg ACR Lag 2": float(match.group(2)),
                        "Avg ACR Lag 3": float(match.group(3)),
                        "Total Rules Broken": int(match.group(4)),
                        "Rules Broken per Vistaar": float(match.group(5)),
                    }
                )

        p_values_list = []
        for lag in [1, 2, 3]:
            matches = re.findall(
                r"t-test (full vs tags|full vs emphasis|tags vs emphasis|full vs base) \(lag "
                + str(lag)
                + r"\):\s+.*? ([\d.e-]+)",
                section_content,
            )
            for match in matches:
                p_values_list.append(
                    {
                        "Group": f"Lag {lag}",
                        "Comparison": match[0].replace(" vs ", " vs. "),
                        "Raw P-Value": float(match[1]),
                    }
                )

        all_data["Phrase Rules"] = {
            "metrics": pd.DataFrame(metrics_list).set_index("Model"),
            "p_values": pd.DataFrame(p_values_list),
        }
        print(" -> Successfully parsed 'Phrase Rules'.")
    except AttributeError:
        print(" -> Could not find or parse 'Phrase Rules' section.")

    # --- Section 2: Pattern Distribution ---
    try:
        section_content = re.search(
            r"--- Ablative Evaluation II - Pattern Distribution ---(.*?)--- Ablative Evaluation III",
            log_content,
            re.DOTALL,
        ).group(1)
        metrics_list = [
            {
                "Model": name,
                "Num Errors": float(
                    re.search(
                        re.escape(desc) + r".*?Num errors\s+([\d.e-]+)",
                        section_content,
                        re.DOTALL,
                    ).group(1)
                ),
            }
            for desc, name in model_name_map.items()
        ]

        p_values_list = []
        matches = re.findall(
            r"Full Model vs\. (Base Model|Tags-Only|Emphasis-Only) p-value: ([\d.e-]+)",
            section_content,
        )
        for match in matches:
            p_values_list.append(
                {"Comparison": f"Full vs. {match[0]}", "Raw P-Value": float(match[1])}
            )

        all_data["Pattern Distribution"] = {
            "metrics": pd.DataFrame(metrics_list).set_index("Model"),
            "p_values": pd.DataFrame(p_values_list),
        }
        print(" -> Successfully parsed 'Pattern Distribution'.")
    except (AttributeError, TypeError):
        print(" -> Could not find or parse 'Pattern Distribution' section.")

    # --- Section 3: Vocab Size ---
    try:
        section_content = re.search(
            r"--- Ablative Evaluation III - Vocab Size ---(.*?)(\[array|--- Structural Fidelity Evaluation ---)",
            log_content,
            re.DOTALL,
        ).group(1)
        metrics_list = []
        for desc, name in model_name_map.items():
            match = re.search(
                re.escape(desc)
                + r".*?Averge phrase` length\s+([\d.e-]+).*?Average vocab size\s+([\d.e-]+)",
                section_content,
                re.DOTALL,
            )
            if match:
                metrics_list.append(
                    {
                        "Model": name,
                        "Avg Phrase Length": float(match.group(1)),
                        "Avg Vocab Size": float(match.group(2)),
                    }
                )

        p_values_list = []
        matches = re.findall(
            r"([\w\s-]+? vs\. [\w\s-]+?) p-value: ([\d.e-]+)", section_content
        )
        for match in matches:
            p_values_list.append(
                {
                    "Comparison": match[0].strip().replace(" Model", ""),
                    "Raw P-Value": float(match[1]),
                }
            )

        all_data["Vocab Size"] = {
            "metrics": pd.DataFrame(metrics_list).set_index("Model"),
            "p_values": pd.DataFrame(p_values_list),
        }
        print(" -> Successfully parsed 'Vocab Size'.")
    except (AttributeError, TypeError):
        print(" -> Could not find or parse 'Vocab Size' section.")

    # --- Section 4: Structural Fidelity ---
    try:
        section_content = re.search(
            r"--- Structural Fidelity Evaluation ---(.*)", log_content, re.DOTALL
        ).group(1)
        metrics_list = []
        model_order = ["Full", "Tags-Only", "Emphasis-Only", "Base"]

        model_blocks = re.split(r"Running with", section_content)
        for i, model_name in enumerate(model_order):
            if (i + 1) < len(model_blocks):
                block = model_blocks[i + 1]
                match = re.search(r"array\((.*?)\)", block, re.DOTALL)
                if match:
                    array_content = match.group(1)
                    # Use regex to find all numbers (including scientific notation) in the array string
                    cleaned_values = re.findall(r"[\d.e-]+", array_content)

                    row_data = {"Model": model_name}
                    for idx, value in enumerate(cleaned_values):
                        # Name columns 'Note 0', 'Note 1', etc., to align with p-value groups
                        row_data[f"Note {idx}"] = float(value)
                    metrics_list.append(row_data)

        p_values_list = []
        matches = re.findall(
            r"note (\d+) (full vs base|full vs tags|full vs emphasis|tags vs emphasis|tags vs base|emphasis vs base): ([\d.e-]+|nan)",
            section_content,
        )
        for match in matches:
            if match[2] != "nan":
                p_values_list.append(
                    {
                        "Group": f"Note {match[0]}",
                        "Comparison": match[1].replace(" vs ", " vs. "),
                        "Raw P-Value": float(match[2]),
                    }
                )
            else:
                p_values_list.append(
                    {
                        "Group": f"Note {match[0]}",
                        "Comparison": match[1].replace(" vs ", " vs. "),
                        "Raw P-Value": pd.NA,
                    }
                )

        # Create a wide DataFrame where each note value is a column
        df_metrics = pd.DataFrame(metrics_list)
        if not df_metrics.empty:
            df_metrics = df_metrics.set_index("Model")

        all_data["Structural Fidelity"] = {
            "metrics": df_metrics,
            "p_values": pd.DataFrame(p_values_list),
        }
        print(" -> Successfully parsed 'Structural Fidelity' into columns.")
    except (AttributeError, IndexError):
        print(" -> Could not find or parse 'Structural Fidelity' section.")

    return all_data


def apply_bonferroni_correction(df):
    """Applies Bonferroni correction to a p-value DataFrame."""
    if df is None or df.empty:
        return df

    group_col = "Group" if "Group" in df.columns else None

    if group_col:
        df["Correction Factor (N)"] = df.groupby(group_col)["Raw P-Value"].transform(
            "count"
        )
    else:
        df["Correction Factor (N)"] = len(df)

    df["Corrected P-Value"] = (df["Raw P-Value"] * df["Correction Factor (N)"]).clip(
        upper=1.0
    )
    return df


def write_formatted_excel(all_data, filename="evaluation_report_by_test.xlsx"):
    """
    Writes all parsed data to a beautifully formatted, multi-sheet Excel file.
    """
    print(f"\nWriting data to '{filename}'...")
    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        workbook = writer.book
        title_format = workbook.add_format(
            {
                "bold": True,
                "font_size": 14,
                "bg_color": "#4F81BD",
                "font_color": "white",
                "valign": "vcenter",
            }
        )
        header_format = workbook.add_format(
            {"bold": True, "bg_color": "#DCE6F1", "border": 1, "text_wrap": True}
        )

        for sheet_name, data in all_data.items():
            worksheet = workbook.add_worksheet(sheet_name)
            current_row = 0

            df_metrics = data.get("metrics")
            if df_metrics is not None and not df_metrics.empty:
                worksheet.merge_range(
                    current_row,
                    0,
                    current_row,
                    len(df_metrics.columns),
                    f"{sheet_name}: Raw Metrics",
                    title_format,
                )
                worksheet.set_row(current_row, 20)
                current_row += 2

                df_metrics.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    startrow=current_row,
                    header=True,
                    index=True,
                )
                for col_num, value in enumerate(
                    [df_metrics.index.name] + list(df_metrics.columns)
                ):
                    worksheet.write(current_row, col_num, value, header_format)

                current_row += len(df_metrics) + 3

            df_p_values = apply_bonferroni_correction(data.get("p_values"))
            if df_p_values is not None and not df_p_values.empty:
                worksheet.merge_range(
                    current_row,
                    0,
                    current_row,
                    len(df_p_values.columns) - 1,
                    f"{sheet_name}: P-Values (Bonferroni Corrected)",
                    title_format,
                )
                worksheet.set_row(current_row, 20)
                current_row += 2

                df_p_values.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    startrow=current_row,
                    header=True,
                    index=False,
                )
                for col_num, value in enumerate(df_p_values.columns):
                    worksheet.write(current_row, col_num, value, header_format)

            worksheet.autofit()
            worksheet.set_column(0, 0, 25)

    print(f"✅ Success! Report saved as '{filename}'.")


# --- Main Execution Block ---
if __name__ == "__main__":
    log_file_path = "eval_log_amrit.txt"
    try:
        with open(log_file_path, "r") as f:
            log_content = f.read()

        parsed_data = parse_log_file(log_content)
        if parsed_data:
            write_formatted_excel(parsed_data)
        else:
            print(
                "Could not parse any data from the log file. No report was generated."
            )

    except FileNotFoundError:
        print(f"❌ Error: The file '{log_file_path}' was not found.")
        print("Please make sure the log file is in the same directory as the script.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
