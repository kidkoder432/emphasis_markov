from re import T
import pandas as pd
import plotly.graph_objects as go
from natsort import natsorted


def create_p_value_heatmap(file_path):
    """
    Reads p-value data from a CSV file and creates a Plotly heatmap.

    Args:
        file_path (str): The path to the CSV file.
    """
    try:
        # Load the data, skipping the top rows of raw metrics
        df = pd.read_excel(file_path, skiprows=11, sheet_name="Structural Fidelity")

        # We only need the first 5 columns based on the file structure
        df = df.iloc[:, :5]

        # Rename columns for easier access
        df.columns = [
            "Note",
            "Comparison",
            "Raw_P_Value",
            "Correction_Factor",
            "p_value",
        ]

        # --- Data Cleaning ---
        # Drop rows where essential data is missing
        df.dropna(subset=["Note", "Comparison", "Raw_P_Value"], inplace=True)

        # Ensure p-value is numeric, coercing errors to NaN and then dropping them
        df["Raw_P_Value"] = pd.to_numeric(df["Raw_P_Value"], errors="coerce")
        df.dropna(subset=["Raw_P_Value"], inplace=True)
        # df.reindex(natsorted(df.index))

        # --- P-Value Categorization ---
        def categorize_p_value(p):
            if p < 0.001:
                return "p < 0.001"
            elif 0.001 <= p <= 0.0083:
                return "0.001 <= p <= 0.0083"
            else:
                return "p > 0.0083"

        df["Category"] = df["Raw_P_Value"].apply(categorize_p_value)

        # --- Prepare Data for Heatmap ---
        # Pivot the data to create a matrix for the heatmap
        pivot_df = df.pivot(
            index="Note",
            columns="Comparison",
            values="Raw_P_Value",
        )
        pivot_df = pivot_df.reindex(natsorted(pivot_df.index))

        # Create a corresponding matrix for the text labels (formatted p-values)
        text_df = pivot_df.map(lambda p: f"{p:.2e}" if pd.notna(p) else "")

        # Create a pivot for categories to drive the color
        category_pivot_df = df.pivot(
            index="Note", columns="Comparison", values="Category"
        )
        category_pivot_df = category_pivot_df.reindex(
            natsorted(category_pivot_df.index)
        )

        # Map categories to numerical values for the colorscale
        category_map = {"p > 0.0083": 0, "0.001 <= p <= 0.0083": 1, "p < 0.001": 2}
        color_z = category_pivot_df.map(lambda x: category_map.get(x, -1))

        # --- Create the Plot ---
        fig = go.Figure(
            data=go.Heatmap(
                z=color_z.values,
                x=color_z.columns,
                y=color_z.index,
                text=text_df.values,
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False,
                colorscale=[
                    [0, "#636EFA"],  # Color for p > 0.0083
                    [0.5, "#FFA15A"],  # Color for 0.001 <= p <= 0.0083
                    [1, "#FF5252"],  # Color for p < 0.001
                ],
                colorbar=dict(
                    title="P-Value Significance",
                    tickvals=[0, 1, 2],
                    ticktext=[
                        "p > 0.0083 (Not Significant)",
                        "0.001 <= p <= 0.0083 (Significant)",
                        "p < 0.001 (Highly Significant)",
                    ],
                ),
                xgap=1,
                ygap=1,  # Add gaps between cells for clarity
            )
        )

        fig.update_layout(
            title="Heatmap of P-Values for Structural Fidelity",
            xaxis_title="Model Comparison",
            yaxis_title="Note",
            template="plotly_dark",
            xaxis_nticks=len(color_z.columns),
            yaxis_nticks=len(color_z.index),
        )

        fig.show()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")


# --- Main Execution ---
if __name__ == "__main__":
    # Make sure to place the CSV file in the same directory as the script,
    # or provide the full path to the file.
    file_to_process = "0920_evals_jog.xlsx"
    create_p_value_heatmap(file_to_process)
