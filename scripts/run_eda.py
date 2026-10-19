"""Create reproducible exploratory data analysis figures."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/heart.csv")
    parser.add_argument("--output", default="reports/eda")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)

    missing = df.isna().sum().sort_values(ascending=False)
    missing.to_csv(out / "missing_values.csv", header=["missing_count"])
    df["target"].value_counts().sort_index().rename_axis("target").to_csv(out / "class_balance.csv")

    ax = df["target"].value_counts().sort_index().plot(kind="bar", rot=0)
    ax.set_title("Heart Disease Target Class Balance")
    ax.set_xlabel("Target (0 = absent, 1 = present)")
    ax.set_ylabel("Records")
    plt.tight_layout()
    plt.savefig(out / "class_balance.png", dpi=160)
    plt.close()

    numeric = df.select_dtypes(include="number")
    numeric.hist(figsize=(14, 10), bins=18, edgecolor="black")
    plt.suptitle("Feature Distributions", y=1.01)
    plt.tight_layout()
    plt.savefig(out / "feature_histograms.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 9))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Numeric Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out / "correlation_heatmap.png", dpi=160)
    plt.close()
    print(f"EDA outputs saved in {out.resolve()}")


if __name__ == "__main__":
    main()
