import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Segment order for consistent matrix layout
SEGMENT_ORDER = ["Champions", "Loyal", "At Risk", "Lost"]


# ------------------------------------------------------------------
# 1. COMPUTE MONTHLY RFM SEGMENTS
# ------------------------------------------------------------------

def assign_segment(score):
    """Maps RFM total score to a segment label."""
    if score >= 12:
        return "Champions"
    elif score >= 9:
        return "Loyal"
    elif score >= 6:
        return "At Risk"
    else:
        return "Lost"


def compute_monthly_segments(df):
    """
    Computes each customer's RFM segment for every month they were active.

    Instead of one final RFM snapshot, we compute RFM scores month by month.
    For each month M, we look at all transactions up to and including month M
    and compute recency/frequency/monetary relative to the end of month M.

    This gives us a time series of segments per customer — essential for
    computing transitions.

    Args:
        df (pd.DataFrame): Cleaned transaction dataframe

    Returns:
        pd.DataFrame: Columns [CustomerID, Month, Segment]
    """

    df = df.copy()
    df["Month"] = df["InvoiceDate"].dt.to_period("M")

    months = sorted(df["Month"].unique())
    records = []

    for month in months:
        # Only consider transactions up to this month
        subset = df[df["Month"] <= month]

        reference_date = month.to_timestamp(how="end")

        rfm = subset.groupby("CustomerID").agg(
            recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            frequency=("Invoice", "nunique"),
            monetary=("TotalPrice", "sum")
        )

        # Score each dimension into quintiles (1-5)
        rfm["R"] = pd.qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop")
        rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5], duplicates="drop")
        rfm["M"] = pd.qcut(rfm["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5], duplicates="drop")

        rfm["RFM_Score"] = rfm["R"].astype(int) + rfm["F"].astype(int) + rfm["M"].astype(int)
        rfm["Segment"] = rfm["RFM_Score"].apply(assign_segment)
        rfm["Month"] = month

        records.append(rfm[["Month", "Segment"]].reset_index())

    monthly_segments = pd.concat(records, ignore_index=True)
    return monthly_segments


# ------------------------------------------------------------------
# 2. BUILD TRANSITION MATRIX
# ------------------------------------------------------------------

def build_transition_matrix(monthly_segments):
    """
    Builds a Markov transition probability matrix from monthly segment data.

    For each customer, for each consecutive pair of months, we record:
        (segment in month T) → (segment in month T+1)

    We then count all transitions and normalise each row to get probabilities.

    Args:
        monthly_segments (pd.DataFrame): Output of compute_monthly_segments()

    Returns:
        pd.DataFrame: Transition probability matrix [SEGMENT_ORDER x SEGMENT_ORDER]
    """

    # Sort so consecutive months are adjacent
    monthly_segments = monthly_segments.sort_values(["CustomerID", "Month"])

    # For each customer, shift segment by one month to get "next segment"
    monthly_segments["Next_Segment"] = monthly_segments.groupby("CustomerID")["Segment"].shift(-1)
    monthly_segments["Next_Month"] = monthly_segments.groupby("CustomerID")["Month"].shift(-1)

    # Keep only consecutive month pairs (no gaps)
    monthly_segments["Month_Diff"] = (
        monthly_segments["Next_Month"] - monthly_segments["Month"]
    ).apply(lambda x: x.n if pd.notna(x) else None)

    transitions = monthly_segments[
        (monthly_segments["Month_Diff"] == 1) &
        monthly_segments["Next_Segment"].notna()
    ].copy()

    # Count transitions
    counts = (
        transitions.groupby(["Segment", "Next_Segment"])
        .size()
        .reset_index(name="count")
        .pivot(index="Segment", columns="Next_Segment", values="count")
        .reindex(index=SEGMENT_ORDER, columns=SEGMENT_ORDER)
        .fillna(0)
    )

    # Normalise rows to get probabilities
    transition_matrix = counts.div(counts.sum(axis=1), axis=0) * 100

    print("=== Markov Transition Matrix (%) ===")
    print(transition_matrix.round(1))

    return transition_matrix


# ------------------------------------------------------------------
# 3. VISUALISE AS HEATMAP
# ------------------------------------------------------------------

def plot_transition_heatmap(transition_matrix, save_path=None):
    """
    Plots the Markov transition matrix as an annotated heatmap.

    Rows = current segment (FROM)
    Columns = next segment (TO)
    Cell value = % probability of moving from row segment to column segment

    High values on the diagonal = segments are stable (customers stay)
    High values off-diagonal = customers are moving between segments

    Args:
        transition_matrix (pd.DataFrame): Output of build_transition_matrix()
        save_path (str): Optional path to save the figure

    Returns:
        fig: matplotlib Figure
    """

    fig, ax = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        transition_matrix,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",          # red = low retention, green = high retention
        vmin=0,
        vmax=100,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Transition Probability (%)"}
    )

    ax.set_title(
        "Customer Segment Transition Matrix (Markov Chain)\n"
        "Row = current segment | Column = next month's segment",
        fontsize=13, pad=15
    )
    ax.set_xlabel("Next Month's Segment", fontsize=11)
    ax.set_ylabel("Current Segment", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ------------------------------------------------------------------
# 4. STEADY STATE — where do customers end up long-term?
# ------------------------------------------------------------------

def compute_steady_state(transition_matrix):
    """
    Computes the steady-state distribution of the Markov chain.

    The steady state answers: "If we ran this chain forever, what fraction
    of customers would be in each segment in the long run?"

    Mathematically, it's the eigenvector of the transition matrix
    corresponding to eigenvalue 1. In practice we solve: π * P = π

    Args:
        transition_matrix (pd.DataFrame): Probability matrix (rows sum to 100)

    Returns:
        pd.Series: Steady-state probabilities per segment
    """

    P = (transition_matrix / 100).values  # convert % back to probabilities

    # Power method: multiply any initial distribution by P repeatedly
    # until it converges — this gives the steady state
    pi = np.ones(len(SEGMENT_ORDER)) / len(SEGMENT_ORDER)  # start uniform
    for _ in range(1000):
        pi_new = pi @ P
        if np.allclose(pi, pi_new, atol=1e-8):
            break
        pi = pi_new

    steady = pd.Series(pi * 100, index=SEGMENT_ORDER)
    print("\n=== Steady-State Distribution (long-run %) ===")
    print(steady.round(2))
    return steady


# ------------------------------------------------------------------
# 5. MAIN
# ------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_data.csv", parse_dates=["InvoiceDate"])

    print("Computing monthly segments (this takes a moment)...")
    monthly_segments = compute_monthly_segments(df)

    transition_matrix = build_transition_matrix(monthly_segments)

    fig = plot_transition_heatmap(transition_matrix, save_path="data/markov_heatmap.png")
    plt.show()
    print("Saved to data/markov_heatmap.png")

    steady = compute_steady_state(transition_matrix)
