from pathlib import Path
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

FEATURES = ["age", "ejection_fraction", "serum_creatinine", "serum_sodium"]

def hierarchical_clustering(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sex_value, label in [(0, "female"), (1, "male")]:
        subset = df[df["sex"] == sex_value][FEATURES].dropna()
        if len(subset) < 2:
            continue
        z = linkage(subset, method="ward")
        plt.figure(figsize=(10,6))
        dendrogram(z, no_labels=True)
        plt.title(f"Hierarchical Clustering — {label.title()}")
        plt.tight_layout()
        plt.savefig(output_dir / f"dendrogram_{label}.png", dpi=150)
        plt.close()
