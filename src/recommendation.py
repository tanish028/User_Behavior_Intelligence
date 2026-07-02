import pandas as pd
import numpy as np


NOISE = {"POSTAGE", "MANUAL", "DOTCOM POSTAGE", "CRUK COMMISSION"}


def recommend_popular(df, n=5):
    """Returns top N products by total revenue."""
    df = df[~df["Description"].str.upper().isin(NOISE)]
    top = df.groupby("Description")["TotalPrice"].sum()
    return top.sort_values(ascending=False).head(n)


def recommend_for_customer(customer_id, df, rfm, n=5):
    """Return top N products for customer based on cluster popularity."""
    customer_id = int(customer_id)
    customer_cluster = rfm.loc[customer_id, "Cluster_Name"]
    same_cluster = rfm[rfm["Cluster_Name"] == customer_cluster].index
    cluster_df = df[df["CustomerID"].isin(same_cluster)]
    cluster_df = cluster_df[~cluster_df["Description"].str.upper().isin(NOISE)]
    top = cluster_df.groupby("Description")["TotalPrice"].sum()
    return top.sort_values(ascending=False).head(n)


def evaluate_recommendations(df, rfm, k=10, sample_size=200, random_state=42):
    """
    Evaluates recommendation quality with Precision@K and Recall@K.

    Method: leave-one-out per customer.
    For each sampled customer:
      1. Hold out their last invoice date as the test set.
      2. Generate K recommendations from their remaining history.
      3. Check how many recommended items appear in the held-out invoice.

    Precision@K = hits / K
    Recall@K    = hits / total items the customer actually bought in test

    Args:
        df          : cleaned transactions DataFrame
        rfm         : RFM DataFrame with Cluster_Name
        k           : number of recommendations to evaluate
        sample_size : customers to sample (full eval is slow)

    Returns:
        dict with Precision@K, Recall@K, n_evaluated
    """
    eligible = (
        df.groupby("CustomerID")["InvoiceDate"]
        .nunique()
        .loc[lambda s: s >= 2]
        .index
    )
    eligible = [c for c in eligible if c in rfm.index]

    rng = np.random.default_rng(random_state)
    sample = rng.choice(eligible, size=min(sample_size, len(eligible)), replace=False)

    precisions, recalls = [], []

    for cid in sample:
        cust_df   = df[df["CustomerID"] == cid]
        last_date = cust_df["InvoiceDate"].max()
        test_df   = cust_df[cust_df["InvoiceDate"] == last_date]
        train_df  = cust_df[cust_df["InvoiceDate"] <  last_date]

        if train_df.empty or test_df.empty:
            continue

        actual = set(
            test_df["Description"].dropna()
            .loc[lambda s: ~s.str.upper().isin(NOISE)]
            .unique()
        )
        if not actual:
            continue

        df_train = df[~((df["CustomerID"] == cid) & (df["InvoiceDate"] == last_date))]
        try:
            recs    = recommend_for_customer(cid, df_train, rfm, n=k)
            rec_set = set(recs.index.tolist()[:k])
        except Exception:
            continue

        if not rec_set:
            continue

        hits = len(actual & rec_set)
        precisions.append(hits / k)
        recalls.append(hits / len(actual))

    return {
        f"Precision@{k}": round(float(np.mean(precisions)), 4) if precisions else 0.0,
        f"Recall@{k}":    round(float(np.mean(recalls)),    4) if recalls    else 0.0,
        "n_evaluated":    len(precisions),
    }


if __name__ == "__main__":
    cleaned_data = pd.read_csv("data/cleaned_data.csv", parse_dates=["InvoiceDate"])
    rfm = pd.read_csv("data/rfm_data.csv", index_col="CustomerID")

    print("=== Popular Products ===")
    print(recommend_popular(cleaned_data))

    for cluster_name in rfm["Cluster_Name"].unique():
        customer_id = rfm[rfm["Cluster_Name"] == cluster_name].index[0]
        print(f"\n=== {cluster_name} (Customer {customer_id}) ===")
        print(recommend_for_customer(customer_id, cleaned_data, rfm))

    print("\n=== Recommendation Metrics ===")
    metrics = evaluate_recommendations(cleaned_data, rfm, k=10, sample_size=100)
    print(metrics)
