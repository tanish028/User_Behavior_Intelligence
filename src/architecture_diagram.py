"""
architecture_diagram.py
Generates a pipeline architecture diagram for the User Behavior Intelligence project.
Saves to data/architecture_diagram.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

C_DATA   = "#DBEAFE"
C_PROC   = "#D1FAE5"
C_MODEL  = "#EDE9FE"
C_OUTPUT = "#FEF3C7"
C_DASH   = "#FECACA"
C_BORDER = "#334155"
C_ARROW  = "#64748B"


def _box(ax, x, y, w, h, label, sublabel="", color=C_DATA, fontsize=9):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        linewidth=1.2,
        edgecolor=C_BORDER,
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y + (0.018 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=C_BORDER, zorder=4)
    if sublabel:
        ax.text(x, y - 0.022, sublabel,
                ha="center", va="center", fontsize=7.5,
                color="#475569", style="italic", zorder=4)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.3, mutation_scale=12),
        zorder=2,
    )


def generate_diagram(save_path=None):
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#F8FAFC")

    y_raw, y_clean, y_rfm = 0.92, 0.78, 0.64
    y_models, y_out, y_dash = 0.46, 0.28, 0.10
    bw, bh, bh_s = 0.17, 0.09, 0.07

    # Row 1 - Raw data
    _box(ax, 0.50, y_raw, 0.30, bh,
         "Raw Data", "Online Retail II Excel  -  1.06M rows", color=C_DATA)

    # Row 2 - Cleaning
    _arrow(ax, 0.50, y_raw - bh/2, 0.50, y_clean + bh/2)
    _box(ax, 0.50, y_clean, 0.30, bh,
         "Data Cleaning", "data_cleaning.py  -  797K clean rows", color=C_PROC)

    # Row 3 - RFM
    _arrow(ax, 0.50, y_clean - bh/2, 0.50, y_rfm + bh/2)
    _box(ax, 0.50, y_rfm, 0.38, bh,
         "RFM Feature Engineering",
         "analytics.py  -  Recency / Frequency / Monetary  -  Quintile scores 1-5",
         color=C_PROC)

    # Row 4 - Models
    model_xs = [0.13, 0.37, 0.63, 0.87]
    model_labels = [
        ("K-Means\nClustering",    "segmentation.py"),
        ("XGBoost\nChurn Model",   "churn_model.py"),
        ("Markov Chain\nAnalysis", "markov_analysis.py"),
        ("Cohort\nRetention",      "cohort_analysis.py"),
    ]
    for mx, (lbl, sub) in zip(model_xs, model_labels):
        rfm_x = max(0.31, min(0.69, mx))
        _arrow(ax, rfm_x, y_rfm - bh/2, mx, y_models + bh_s/2)
        _box(ax, mx, y_models, bw, bh_s, lbl, sub, color=C_MODEL, fontsize=8.5)

    # Row 5 - Outputs
    output_labels = [
        ("3 Segments\n+ PCA Plot",        "Silhouette: 0.528"),
        ("Churn Prob + SHAP\nAUC: 0.778", "F-Score drives 91%"),
        ("Steady-State\nDistribution",     "Lost: 25.9% long-run"),
        ("Retention\nHeatmap",             "Month-1: 22.5%"),
    ]
    for mx, (lbl, sub) in zip(model_xs, output_labels):
        _arrow(ax, mx, y_models - bh_s/2, mx, y_out + bh_s/2)
        _box(ax, mx, y_out, bw, bh_s, lbl, sub, color=C_OUTPUT, fontsize=8)

    # Recommendations side branch
    rec_y = (y_out + y_dash) / 2
    _arrow(ax, 0.13, y_out - bh_s/2, 0.13, rec_y + bh_s/2)
    _box(ax, 0.13, rec_y, bw, bh_s,
         "Recommendations", "recommendation.py\nPrecision@10 / Recall@10",
         color=C_OUTPUT, fontsize=8)

    # Row 6 - Dashboard
    for mx in model_xs:
        _arrow(ax, mx, y_out - bh_s/2, mx, y_dash + bh/2 + 0.005)
    _arrow(ax, 0.13, rec_y - bh_s/2, 0.13, y_dash + bh/2 + 0.005)
    _box(ax, 0.50, y_dash, 0.80, bh,
         "Streamlit Dashboard",
         "6 tabs  -  Business insights  -  Live churn predictor  -  SHAP explainability",
         color=C_DASH, fontsize=10)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=C_DATA,   edgecolor=C_BORDER, label="Data layer",  linewidth=0.8),
        mpatches.Patch(facecolor=C_PROC,   edgecolor=C_BORDER, label="Processing",  linewidth=0.8),
        mpatches.Patch(facecolor=C_MODEL,  edgecolor=C_BORDER, label="ML models",   linewidth=0.8),
        mpatches.Patch(facecolor=C_OUTPUT, edgecolor=C_BORDER, label="Outputs",     linewidth=0.8),
        mpatches.Patch(facecolor=C_DASH,   edgecolor=C_BORDER, label="Dashboard",   linewidth=0.8),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.03), ncol=5,
              frameon=True, fontsize=8.5, handlelength=1.4, edgecolor="#CBD5E1")

    ax.set_title("User Behavior Intelligence - System Architecture",
                 fontsize=14, fontweight="bold", color=C_BORDER, pad=6)
    plt.tight_layout(pad=0.4)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print("Architecture diagram saved ->", save_path)

    return fig


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "data", "architecture_diagram.png")
    generate_diagram(save_path=os.path.abspath(out))
