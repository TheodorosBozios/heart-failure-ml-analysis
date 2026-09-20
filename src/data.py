from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = [
    "age","anaemia","creatinine_phosphokinase","diabetes","ejection_fraction",
    "high_blood_pressure","platelets","serum_creatinine","serum_sodium",
    "sex","smoking","time","DEATH_EVENT"
]

def load_data(path):
    df = pd.read_csv(path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df

def save_data_checks(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.describe(include="all").T.to_csv(output_dir / "descriptive_statistics.csv")
    df.isna().sum().rename("missing_values").to_csv(output_dir / "missing_values.csv")
    df.corr(numeric_only=True).to_csv(output_dir / "correlation_matrix.csv")
