"""
clv_model.py
Customer Lifetime Value (CLV) modelling.

Formula:
    CLV = avg_monthly_spend x expected_months_remaining

Where:
    avg_monthly_spend      = total_spend / months_active  (historical)
    expected_months_remaining = derived from the XGBoost churn probability

    If annual churn probability = p, then monthly churn probability
        p_monthly = 1 - (1 - p)^(1/12)
    Expected remaining lifetime under a geometric model:
        E[months] = 1 / p_monthly

This ties the CLV model directly to the churn classifier, creating a
coherent story: the same XGBoost model that flags at-risk customers also
informs how much revenue we expect to lose if they churn.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# CLV computation
# ---------------------------------------------------------------------------

def compute_clv(df: pd.DataFrame, rfm: pd.DataFrame, model) -> pd.DataFrame:
    """
    Compute predicted CLV for every customer in rfm.

    Args:
        df    : cleaned transactions DataFrame
        rfm   : RFM DataFrame with F-Score, M-Score, Segment, Cluster_Name
        model : trained XGBClassifier (from churn_model.train_churn_model)

    Returns:
        clv_df : DataFrame indexed by CustomerID with columns:
                   total_spend, months_active, avg_monthly_spend,
                   churn_prob_annual, expected_months, predicted_clv,
                   Segment, Cluster_Name
    """
    # Per-customer historical metrics
    stats = (
        df[df["CustomerID"].notna()]
        .groupby("CustomerID")
        .agg(
            total_spend   =("TotalPrice",   "sum"),
            first_purchase=("InvoiceDate",  "min"),
            last_purchase =("InvoiceDate",  "max"),
            num_orders    =("Invoice",       "nunique"),
        )
    )

    # Months active = span from first to last purchase
    stats["months_active"] = (
        (stats["last_purchase"] - stats["first_purchase"]).dt.days / 30.44
    ).clip(lower=1.0)

    stats["avg_monthly_spend"] = stats["total_spend"] / stats["months_active"]

    # Only keep customers who exist in rfm
    stats = stats.loc[stats.index.isin(rfm.index)]

    # Annual churn probability from XGBoost (F-Score + M-Score only)
    features = rfm.loc[stats.index, ["F-Score", "M-Score"]].astype(float)
    churn_prob_annual = pd.Series(
        model.predict_proba(features)[:, 1],
        index=stats.index,
    )

    # Convert annual churn prob to monthly, then expected months remaining
    p_monthly        = 1 - (1 - churn_prob_annual.clip(0.01, 0.99)) ** (1 / 12)
    expected_months  = (1 / p_monthly).clip(upper=60)  # cap at 5 years

    clv_df = stats[["total_spend", "months_active", "avg_monthly_spend"]].copy()
    clv_df["churn_prob_annual"]  = churn_prob_annual.values
    clv_df["expected_months"]    = expected_months.values
    clv_df["predicted_clv"]      = (
        clv_df["avg_monthly_spend"] * clv_df["expected_months"]
    )
    clv_df["Segment"]      = rfm.loc[clv_df.index, "Segment"]
    clv_df["Cluster_Name"] = rfm.loc[clv_df.index, "Cluster_Name"]

    return clv_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

BG     = "#F8FAFC"
BORDER = "#334155"
SEG_COLORS = {
    "Champions":   "#2563EB",
    "Loyal":       "#10B981",
    "At Risk":     "#F59E0B",
    "Lost":        "#EF4444",
}


def plot_clv_by_segment(clv_df: pd.DataFrame, save_path=None):
    """
    Two-panel figure:
    Left  — average predicted CLV by RFM segment (bar chart)
    Right — CLV distribution box-plot by segment
    """
    seg_order = ["Champions", "Loyal", "At Risk", "Lost"]
    seg_order = [s for s in seg_order if s in clv_df["Segment"].unique()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(BG)

    # Left: average CLV per segment
    avg_clv = (
        clv_df.groupby("Segment")["predicted_clv"]
        .mean()
        .reindex(seg_order)
    )
    colors = [SEG_COLORS.get(s, "#94A3B8") for s in seg_order]
    bars = ax1.barh(seg_order, avg_clv.values, color=colors,
                    edgecolor="white", height=0.55)
    for bar, v in zip(bars, avg_clv.values):
        ax1.text(bar.get_width() + avg_clv.max() * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"PS{v:,.0f}", va="center", fontsize=9, color=BORDER)
    ax1.set_facecolor(BG)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.spines[["left", "bottom"]].set_color("#CBD5E1")
    ax1.set_title("Average Predicted CLV by Segment",
                  fontsize=11, fontweight="bold", color=BORDER, pad=8)
    ax1.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"PS{v/1000:.0f}K")
    )
    ax1.set_xlim(0, avg_clv.max() * 1.25)
    ax1.tick_params(colors="#475569", labelsize=9)
    ax1.grid(axis="x", color="#E2E8F0", lw=0.7, linestyle="--")
    ax1.set_axisbelow(True)

    # Right: distribution
    data_by_seg = [
        clv_df.loc[clv_df["Segment"] == s, "predicted_clv"].values
        for s in seg_order
    ]
    bp = ax2.boxplot(data_by_seg, vert=True, patch_artist=True,
                     medianprops=dict(color="white", lw=2),
                     whiskerprops=dict(color="#94A3B8"),
                     capprops=dict(color="#94A3B8"),
                     flierprops=dict(marker=".", color="#94A3B8",
                                     markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax2.set_xticks(range(1, len(seg_order) + 1))
    ax2.set_xticklabels(seg_order, fontsize=9)
    ax2.set_facecolor(BG)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.spines[["left", "bottom"]].set_color("#CBD5E1")
    ax2.set_title("CLV Distribution by Segment",
                  fontsize=11, fontweight="bold", color=BORDER, pad=8)
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"PS{v/1000:.0f}K")
    )
    ax2.tick_params(colors="#475569", labelsize=9)
    ax2.grid(axis="y", color="#E2E8F0", lw=0.7, linestyle="--")
    ax2.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


def plot_clv_vs_churn(clv_df: pd.DataFrame, save_path=None):
    """
    Scatter plot: churn probability vs predicted CLV, coloured by segment.
    Shows the trade-off: high CLV + high churn = highest priority for retention.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for seg, grp in clv_df.groupby("Segment"):
        ax.scatter(
            grp["churn_prob_annual"],
            grp["predicted_clv"],
            c=SEG_COLORS.get(seg, "#94A3B8"),
            alpha=0.45, s=18, label=seg, edgecolors="none",
        )

    # Quadrant lines
    median_churn = clv_df["churn_prob_annual"].median()
    median_clv   = clv_df["predicted_clv"].median()
    ax.axvline(median_churn, color="#CBD5E1", lw=1, linestyle="--")
    ax.axhline(median_clv,   color="#CBD5E1", lw=1, linestyle="--")

    # Quadrant labels
    ax.text(median_churn * 0.5, clv_df["predicted_clv"].quantile(0.92),
            "Low risk\nHigh value", ha="center", fontsize=8,
            color="#10B981", fontweight="bold")
    ax.text(median_churn * 1.5, clv_df["predicted_clv"].quantile(0.92),
            "High risk\nHigh value\n(PRIORITY)", ha="center", fontsize=8,
            color="#EF4444", fontweight="bold")
    ax.text(median_churn * 0.5, clv_df["predicted_clv"].quantile(0.05),
            "Low risk\nLow value", ha="center", fontsize=8, color="#94A3B8")
    ax.text(median_churn * 1.5, clv_df["predicted_clv"].quantile(0.05),
            "High risk\nLow value", ha="center", fontsize=8, color="#94A3B8")

    ax.set_xlabel("Annual churn probability (from XGBoost)", fontsize=10, color="#475569")
    ax.set_ylabel("Predicted CLV", fontsize=10, color="#475569")
    ax.set_title("CLV vs Churn Risk — Who to prioritise for retention?",
                 fontsize=12, fontweight="bold", color=BORDER, pad=10)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"PS{v/1000:.0f}K")
    )
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%")
    )
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    ax.tick_params(colors="#475569", labelsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from churn_model import create_churn_label, train_churn_model
    df  = pd.read_csv("data/cleaned_data.csv", parse_dates=["InvoiceDate"])
    rfm = pd.read_csv("data/rfm_data.csv", index_col="CustomerID")
    rfm_l = create_churn_label(rfm)
    model, _, _, _ = train_churn_model(rfm_l)

    clv_df = compute_clv(df, rfm, model)
    print(clv_df.sort_values("predicted_clv", ascending=False).head(10).to_string())
    print(f"\nAvg CLV by segment:\n{clv_df.groupby('Segment')['predicted_clv'].mean().sort_values(ascending=False)}")

    plot_clv_by_segment(clv_df)
    plot_clv_vs_churn(clv_df)
    plt.show()
