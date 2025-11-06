# modules/visualize_data.py
import os
import math
import json
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
    """
    Summarize dataset: shapes, types, missing values, cardinality, basic stats.
    Saves a markdown summary and returns a dict with key facts.
    """
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
    lines.append("## Head (first 5 rows)\n")
    lines.append(df.head(5).to_markdown(index=False))

    _save_text(os.path.join(out_dir, f"{name}__profile.md"), "\n".join(lines))
    _save_json(os.path.join(out_dir, f"{name}__profile.json"), summary)
    return summary


def plot_missingness(df: pd.DataFrame, name: str, out_dir: str) -> None:
    _ensure_dir(out_dir)
    missing = df.isna().sum()
    if missing.sum() == 0:
        # nothing to plot
        return
    plt.figure()
    missing.plot(kind="bar")
    plt.title(f"Missing values per column — {name}")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}__missing_per_column.png"), dpi=140)
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
        df[col].plot(kind="hist", bins=bins)
        plt.title(col, fontsize=9)
        plt.xlabel("")
        plt.ylabel("")
    plt.suptitle(f"Numeric distributions (first {len(cols)} of {len(numeric_cols)}) — {name}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, f"{name}__numeric_histograms.png"), dpi=140)
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
        df[col].value_counts(dropna=False).plot(kind="bar")
        plt.title(col, fontsize=9)
        plt.xlabel("")
        plt.ylabel("")
    plt.suptitle(f"Categorical distributions (≤{max_unique} unique) — {name}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, f"{name}__categorical_bars.png"), dpi=140)
    plt.close()


def plot_correlation(df: pd.DataFrame, name: str, out_dir: str, max_cols: int = 80) -> None:
    _ensure_dir(out_dir)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        return
    
    numeric_cols = numeric_cols[:max_cols]
    corr = df[numeric_cols].corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90, fontsize=6)
    plt.yticks(range(len(numeric_cols)), numeric_cols, fontsize=6)
    plt.title(f"Correlation heatmap (first {len(numeric_cols)}) — {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}__corr_heatmap.png"), dpi=160)
    plt.close()


def plot_pca_2d(df: pd.DataFrame,
                name: str,
                out_dir: str,
                label_col: str = "Outcome",
                exclude_cols: Optional[set] = None) -> None:
    """
    PCA on numeric features only; points coloured by label if present.
    """
    exclude = set(exclude_cols or EXCLUDED_COLS_DEFAULT)
    _ensure_dir(out_dir)

    numeric_cols, _ = _split_types(df, exclude)
    if len(numeric_cols) < 2:
        return

    X = df[numeric_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    y = None
    if label_col in df.columns:
        # Align labels with cleaned X
        y = df.loc[X.index, label_col]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(Xs)

    plt.figure(figsize=(7, 6))
    if y is not None:
        # Encode y as integers for colors
        y_codes = pd.Categorical(y).codes
        scatter = plt.scatter(X2[:, 0], X2[:, 1], c=y_codes, s=16)
        # Build legend from categories
        categories = list(pd.Categorical(y).categories)
        handles = [plt.Line2D([0], [0], marker='o', linestyle='', markersize=6) for _ in categories]
        plt.legend(handles, [str(c) for c in categories], title=label_col, loc="best", fontsize=8)
    else:
        plt.scatter(X2[:, 0], X2[:, 1], s=16)

    evr = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    plt.title(f"PCA (2D) — {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}__pca2d.png"), dpi=160)
    plt.close()


# ---------- public API ----------

def visualize_dataset(df: pd.DataFrame,
                      name: str,
                      base_out_dir: str = "results/visualizations",
                      label_col: str = "Outcome",
                      exclude_cols: Optional[set] = None) -> Dict:
    """
    Run a complete profiling + visualization bundle for one dataframe.
    Returns the profile dict (facts about the dataset).
    """
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
    """
    Simple console menu to pick which dataset(s) to visualize.
    Saves images and markdown into results/visualizations/...
    """
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
