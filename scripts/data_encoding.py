import pandas as pd

# Load the previously structured dataset
df = pd.read_csv("formatted_wide.csv")

# ==========================================================
# STEP 1: Fill empty values in
# 'nigerian_adaptation.suggested_markers[2]'
# ==========================================================
col_fill = "nigerian_adaptation.suggested_markers[2]"

if col_fill in df.columns:
    mask = df[col_fill].isna() | (df[col_fill].astype(str).str.strip() == "")
    df.loc[mask, col_fill] = "non standard"

# ==========================================================
# STEP 2: Encode categorical columns into 0 and 1
# ==========================================================
mappings = {
    "behavioral_profile.rating_consistency": {
        "stable": 1,
        "volatile": 0,
    },

    "behavioral_profile.sentiment_bias": {
        "critical": 1,
        "generous": 0,
    },

    "behavioral_profile.verbal_style": {
        "detailed": 1,
        "concise": 0,
        "consice": 0,   # Handles misspelling
    },

    "nigerian_adaptation.suggested_markers[0]": {
        "correct": 1,
        "sha": 0,
    },

    "nigerian_adaptation.suggested_markers[1]": {
        "too much": 1,
        "abeg": 0,
    },

    "nigerian_adaptation.suggested_markers[2]": {
        "standard": 1,
        "non standard": 0,
        "non-standard": 0,
    }
}

# Apply encoding
for col, mapping in mappings.items():
    if col in df.columns:
        normalized = df[col].astype(str).str.strip().str.lower()
        df[col] = normalized.map(mapping)

# ==========================================================
# STEP 3: Save the processed dataset
# ==========================================================
df.to_csv("formatted_wide_encoded.csv", index=False)
df.to_excel("formatted_wide_encoded.xlsx", index=False)

# ==========================================================
# STEP 4: Verify results
# ==========================================================
print("Dataset processed successfully.")
print("Shape:", df.shape)
print("Files created:")
print("- formatted_wide_encoded.csv")
print("- formatted_wide_encoded.xlsx")

# Optional: Display unique values after encoding
for col in mappings.keys():
    if col in df.columns:
        print(f"{col}: {sorted(df[col].dropna().unique())}")
