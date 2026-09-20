from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def make_eda_plots(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x="DEATH_EVENT", y="age")
    plt.title("Age by Outcome")
    plt.tight_layout()
    plt.savefig(output_dir / "age_by_outcome.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6,5))
    sns.countplot(data=df, x="DEATH_EVENT")
    plt.title("Outcome Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "outcome_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()
