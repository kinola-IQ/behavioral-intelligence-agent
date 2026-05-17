"""Clean and normalize input records."""
import json
import pandas as pd

# Path to your JSON or Jupyter Notebook (.ipynb) file
input_file = r"path to input file"

# Output files
excel_output = r"path to excel output"
csv_output = r"path to csv output"


def flatten_json(data, parent_key=""):
    """
    Recursively flatten nested JSON objects and arrays into:
    key.path -> value
    """
    items = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            items.extend(flatten_json(value, new_key))

    elif isinstance(data, list):
        for i, value in enumerate(data):
            new_key = f"{parent_key}[{i}]"
            items.extend(flatten_json(value, new_key))

    else:
        # Convert None to blank cell in Excel/CSV
        items.append((parent_key, "" if data is None else data))

    return items


def save_linear_to_excel(df, output_path):
    """
    Save flattened data to Excel.
    """
    df.to_excel(output_path, index=False, engine="openpyxl")
    print(f"Excel file created: {output_path}")


def save_linear_to_csv(df, output_path):
    """
    Save flattened data to CSV.
    """
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"CSV file created: {output_path}")
