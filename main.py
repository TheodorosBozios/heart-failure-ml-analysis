import argparse
from pathlib import Path
from src.data import load_data, save_data_checks
from src.eda import make_eda_plots
from src.models import train_models
from src.clustering import hierarchical_clustering

def main():
    parser = argparse.ArgumentParser(description="Heart Failure Clinical Records portfolio analysis")
    parser.add_argument("--data", default="data/heart_failure_clinical_records_dataset.csv")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--show-plots", action="store_true", help="Reserved for future interactive plotting support.")
    args = parser.parse_args()

    df = load_data(args.data)
    save_data_checks(df, args.output)
    make_eda_plots(df, args.output)
    hierarchical_clustering(df, args.output)
    train_models(df, args.output)
    print(f"Analysis complete. Results saved to {Path(args.output).resolve()}")

if __name__ == "__main__":
    main()
