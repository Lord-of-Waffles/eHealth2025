# modules/visualize_data.py
import os
import math
import json
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # <-- new!
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ---------- Global styling ----------
# Set once for all plots
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "savefig.dpi": 200,
    "figure.autolayout": True,
})


# ---------- helpers ----------

EXCLUDED_COLS_DEFAULT = {"Outcome", "PatientID", "CenterID"}

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _split_types(df: pd.DataFrame, exclude: set) -> Tuple[List[str], List[str]]:
    cols = [c for c in df.columns if c not in exclude]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in cols if c not in numeric]
    return numeric, categorical

def _save_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def _save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ---------- core visualizations ----------

def describe_dataset(df: pd.DataFrame,
                     name: str,
                     out_dir: str,
                     label_col: str = "Outcome",
                     exclude_cols: Optional[set] = None) -> Dict:
    exclude = set(exclude_cols or EXCLUDED_COLS_DEFAULT)
    _ensure_dir(out_dir)

    n_rows, n_cols = df.shape
    numeric_cols, categorical_cols = _split_types(df, exclude)

    missing_per_col = df.isna().sum().to_dict()
    cardinality = {c: int(df[c].nunique(dropna=True)) for c in df.columns}

    summary = {
        "dataset": name,
        "rows": int(n_rows),
        "columns": int(n_cols),
        "label_present": label_col in df.columns,
        "excluded_columns": sorted(list(exclude & set(df.columns))),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": missing_per_col,
        "cardinality_by_column": cardinality,
        "head_preview": df.head(5).to_dict(orient="list")
    }

    # Markdown report
    lines = []
    lines.append(f"# Data profile — {name}\n")
    lines.append(f"- Shape: **{n_rows} x {n_cols}**")
    lines.append(f"- Label column present: **{summary['label_present']}**")
    if summary["excluded_columns"]:
        lines.append(f"- Excluded columns (not plotted as features): `{', '.join(summary['excluded_columns'])}`")
    lines.append(f"- Numeric features ({len(numeric_cols)}): `{', '.join(numeric_cols[:15])}{'…' if len(numeric_cols)>15 else ''}`")
    lines.append(f"- Categorical features ({len(categorical_cols)}): `{', '.join(categorical_cols[:15])}{'…' if len(categorical_cols)>15 else ''}`")
    lines.append(f"- Total missing values: **{summary['missing_total']}**\n")
    try:
        lines.append("## Head (first 5 rows)\n")
        lines.append(df.head(5).to_markdown(index=False))
    except ImportError:
        # Fallback if tabulate not installed
        lines.append("## Head (first 5 rows)\n")
        lines.append("```\n" + df.head(5).to_string() + "\n```")

    _save_text(os.path.join(out_dir, f"{name}__profile.md"), "\n".join(lines))
    _save_json(os.path.join(out_dir, f"{name}__profile.json"), summary)
    return summary


def plot_missingness(df: pd.DataFrame, name: str, out_dir: str) -> None:
    _ensure_dir(out_dir)
    missing = df.isna().sum()
    if missing.sum() == 0:
        return

    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return

    plt.figure(figsize=(8, 4))
    missing.plot(kind="bar", color="firebrick")
    plt.title(f"Missing values per column — {name}", fontsize=12)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}__missing_per_column.png"))
    plt.close()


def plot_histograms(df: pd.DataFrame,
                    name: str,
                    out_dir: str,
                    bins: int = 30,
                    max_plots: int = 36) -> None:
    _ensure_dir(out_dir)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return

    cols = numeric_cols[:max_plots]
    n = len(cols)
    rows = math.ceil(n / 6)
    plt.figure(figsize=(18, 3.2 * rows))
    for i, col in enumerate(cols, 1):
        plt.subplot(rows, 6, i)
        sns.histplot(df[col].dropna(), kde=True, bins=bins, color="steelblue")
        plt.title(col, fontsize=9)
        plt.xlabel("")
        plt.ylabel("")
    plt.suptitle(f"Numeric distributions (first {len(cols)} of {len(numeric_cols)}) — {name}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, f"{name}__numeric_histograms.png"))
    plt.close()


def plot_categorical_bars(df: pd.DataFrame,
                          name: str,
                          out_dir: str,
                          max_unique: int = 20,
                          max_plots: int = 24) -> None:
    _ensure_dir(out_dir)
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in cat_cols if df[c].nunique(dropna=True) <= max_unique]
    if not cat_cols:
        return

    cols = cat_cols[:max_plots]
    n = len(cols)
    rows = math.ceil(n / 6)
    plt.figure(figsize=(18, 3.2 * rows))
    for i, col in enumerate(cols, 1):
        plt.subplot(rows, 6, i)
        vc = df[col].value_counts(dropna=False).sort_values(ascending=True)
        vc.plot(kind="barh", color="lightcoral")
        plt.title(col, fontsize=9)
        plt.xlabel("")
        plt.ylabel("")
    plt.suptitle(f"Categorical distributions (≤{max_unique} unique values) — {name}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, f"{name}__categorical_bars.png"))
    plt.close()


def plot_correlation(df: pd.DataFrame, name: str, out_dir: str, max_cols: int = 20) -> None:
    _ensure_dir(out_dir)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        return

    numeric_cols = numeric_cols[:max_cols]
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5
    )
    plt.title(f"Feature Correlation Matrix — {name}", pad=20, fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}__corr_heatmap.png"))
    plt.close()


def plot_pca_2d(df: pd.DataFrame,
                name: str,
                out_dir: str,
                label_col: str = "Outcome",
                exclude_cols: Optional[set] = None) -> None:
    exclude = set(exclude_cols or EXCLUDED_COLS_DEFAULT)
    _ensure_dir(out_dir)

    numeric_cols, _ = _split_types(df, exclude)
    if len(numeric_cols) < 2:
        return

    X = df[numeric_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    y = None
    if label_col in df.columns:
        y = df.loc[X.index, label_col]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(Xs)

    plt.figure(figsize=(7, 6))
    if y is not None:
        y_cat = pd.Categorical(y)
        unique_labels = y_cat.categories
        palette = sns.color_palette("husl", len(unique_labels))
        for i, label in enumerate(unique_labels):
            mask = y == label
            plt.scatter(
                X2[mask, 0], X2[mask, 1],
                label=str(label),
                color=palette[i],
                s=25,
                alpha=0.7,
                edgecolors="w",
                linewidth=0.3
            )
        plt.legend(title=label_col, frameon=True, fontsize=9, title_fontsize=10)
    else:
        plt.scatter(X2[:, 0], X2[:, 1], s=25, alpha=0.7, color="gray", edgecolors="w", linewidth=0.3)

    evr = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({evr[0] * 100:.1f}% variance)", fontsize=11)
    plt.ylabel(f"PC2 ({evr[1] * 100:.1f}% variance)", fontsize=11)
    plt.title(f"PCA (2D Projection) — {name}", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}__pca2d.png"))
    plt.close()


# ---------- public API ----------

def visualize_dataset(df: pd.DataFrame,
                      name: str,
                      base_out_dir: str = "results/visualizations",
                      label_col: str = "Outcome",
                      exclude_cols: Optional[set] = None) -> Dict:
    dataset_dir = os.path.join(base_out_dir, name)
    _ensure_dir(dataset_dir)

    profile = describe_dataset(df, name, dataset_dir, label_col=label_col, exclude_cols=exclude_cols)
    plot_missingness(df, name, dataset_dir)
    plot_histograms(df, name, dataset_dir)
    plot_categorical_bars(df, name, dataset_dir)
    plot_correlation(df, name, dataset_dir)
    plot_pca_2d(df, name, dataset_dir, label_col=label_col, exclude_cols=exclude_cols)

    return profile


def visualize_menu(df_dict: Dict[str, pd.DataFrame]) -> None:
    options = {
        "1": ("clinical", "Clinical"),
        "2": ("ct", "CT"),
        "3": ("pt", "PT"),
        "4": ("all", "All (clinical + CT + PT)")
    }

    print("\n=== Data Visualization ===")
    print("What would you like to visualize?")
    for k, (_, label) in options.items():
        print(f"{k}) {label}")
    choice = input("Enter choice (1-4): ").strip()

    if choice not in options:
        print("Invalid choice.")
        return

    key, label = options[choice]
    if key == "all":
        for k in ("clinical", "ct", "pt"):
            if k in df_dict:
                print(f"\n… Visualizing {k.upper()} …")
                visualize_dataset(df_dict[k], k)
        print("\nSaved visualizations under results/visualizations/<dataset>/")
    else:
        if key not in df_dict:
            print(f"Dataset '{key}' not loaded.")
            return
        visualize_dataset(df_dict[key], key)
        print(f"\nSaved visualizations under results/visualizations/{key}/")