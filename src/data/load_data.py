"""Load raw and external datasets."""
# Load JSON data
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Flatten JSON into key-value pairs
flattened = flatten_json(data)

# Convert to DataFrame with two columns:
# Column A = Path
# Column B = Value
df = pd.DataFrame(flattened, columns=["Path", "Value"])

# Save to both formats
save_linear_to_excel(df, excel_output)
save_linear_to_csv(df, csv_output)

print("Conversion completed successfully.")
