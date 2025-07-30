import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# 1. Load and tidy
# ---------------------------------------------------------------------------

df = pd.read_csv("grand_results_table.csv")

df["Accuracy_pct"] = df["Accuracy"].apply(lambda x: x * 100 if x <= 1 else x)
agg = (
    df.groupby(["Classification", "Model", "Plant"], as_index=False)["Accuracy_pct"].mean()
)

# ---------------------------------------------------------------------------
# 2. Accuracy table (optional export)
# ---------------------------------------------------------------------------

pivot_table = (
    agg.pivot(index="Model", columns=["Classification", "Plant"], values="Accuracy_pct")
    .round(2)
    .sort_index(axis=1)
)
print(pivot_table)  # or pivot_table.to_csv("mean_accuracy_table.csv")

# ---------------------------------------------------------------------------
# 3. Heat‑map generator
# ---------------------------------------------------------------------------

def plot_heatmap(classification: str, out_file: str):
    sub = agg[agg["Classification"] == classification]
    data = sub.pivot(index="Model", columns="Plant", values="Accuracy_pct")

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        linewidths=0.4,
        cbar_kws={"label": "Accuracy (%)"},
    )
    plt.title(f"Accuracy Heat‑map – {classification.capitalize()} Classification", pad=10)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

for cls in ["binary", "detailed", "generation"]:
    plot_heatmap(cls, f"heatmap_{cls}.png")
    print(f"Saved heatmap_{cls}.png")