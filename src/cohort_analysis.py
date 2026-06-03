import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def build_cohort_matrix(df):
    """
    Builds a cohort retention matrix from cleaned transaction data.

    Steps:
    1. Find each customer's first purchase month (their cohort)
    2. For every purchase, compute how many months after their first purchase it was (period)
    3. Count unique customers per (cohort, period)
    4. Divide by cohort size to get retention percentages

    Args:
        df (pd.DataFrame): Cleaned dataframe with CustomerID and InvoiceDate columns

    Returns:
        cohort_pct (pd.DataFrame): Retention matrix as percentages (rows=cohorts, cols=periods)
        cohort_size (pd.Series): Number of customers in each cohort
    """

    df = df.copy()

    # Step 1 — extract the month of each transaction
    df["TransactionMonth"] = df["InvoiceDate"].dt.to_period("M")

    # Step 2 — find each customer's first purchase month (their cohort)
    first_purchase = df.groupby("CustomerID")["TransactionMonth"].min().rename("CohortMonth")
    df = df.merge(first_purchase, on="CustomerID")

    # Step 3 — calculate period: how many months after their first purchase
    df["CohortPeriod"] = (
        df["TransactionMonth"].dt.to_timestamp() - df["CohortMonth"].dt.to_timestamp()
    ).dt.days // 30  # approximate months

    # Step 4 — count unique customers per (cohort, period)
    cohort_matrix = (
        df.groupby(["CohortMonth", "CohortPeriod"])["CustomerID"]
        .nunique()
        .reset_index()
        .pivot(index="CohortMonth", columns="CohortPeriod", values="CustomerID")
    )

    # Step 5 — cohort size = column 0 (everyone was active at period 0)
    cohort_size = cohort_matrix[0]

    # Step 6 — convert to retention percentages
    cohort_pct = cohort_matrix.divide(cohort_size, axis=0) * 100

    return cohort_pct, cohort_size


def plot_cohort_heatmap(cohort_pct, max_periods=12, save_path=None):
    """
    Plots the cohort retention matrix as an annotated heatmap.

    Args:
        cohort_pct (pd.DataFrame): Retention matrix from build_cohort_matrix()
        max_periods (int): Number of months to show (default 12)
        save_path (str): Optional path to save the figure
    """

    # Limit columns to max_periods for readability
    data = cohort_pct.iloc[:, :max_periods]

    # Format index as string for cleaner labels
    data.index = data.index.astype(str)
    data.columns = [f"Month {c}" for c in data.columns]

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(
        data,
        mask=data.isnull(),       # hide empty cells
        annot=True,                # show values in cells
        fmt=".0f",                 # no decimal places
        cmap="YlGnBu",             # yellow-green-blue: low=light, high=dark
        vmin=0,
        vmax=100,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Retention %"}
    )

    ax.set_title("Customer Cohort Retention Analysis\n(% of cohort still active each month)",
                 fontsize=14, pad=15)
    ax.set_xlabel("Months Since First Purchase", fontsize=11)
    ax.set_ylabel("Cohort (First Purchase Month)", fontsize=11)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_data.csv", parse_dates=["InvoiceDate"])

    cohort_pct, cohort_size = build_cohort_matrix(df)

    print("=== Cohort Sizes ===")
    print(cohort_size)
    print("\n=== Retention Matrix (first 6 periods) ===")
    print(cohort_pct.iloc[:, :6].round(1))

    plot_cohort_heatmap(cohort_pct, save_path="data/cohort_heatmap.png")
    plt.show()
    print("\nSaved to data/cohort_heatmap.png")
