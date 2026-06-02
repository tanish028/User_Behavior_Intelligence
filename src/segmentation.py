import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from data_loader import load_data
from data_cleaning import clean_data
from analytics import compute_rfm
from sklearn.decomposition import PCA


def plot_elbow(rfm):

    X = rfm[["recency","frequency","monetary"]]

    k_values = range(1,11)
    inertias  = []

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    silhouette_scores = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        if k > 1:
            silhouette_scores.append(silhouette_score(X_scaled, labels))
        else:
            silhouette_scores.append(None)

    print("Inertias:", inertias)
    print("Silhouette scores (k=2..10):", [round(s, 3) for s in silhouette_scores if s is not None])

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o", color="steelblue")
    plt.title("Elbow Method — Optimal K")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.xticks(range(1, 11))
    plt.tight_layout()
    plt.savefig("data/elbow_plot.png")
    plt.show()


def fit_clusters(rfm,k):


    X = rfm[["recency","frequency","monetary"]]
    # Cap monetary and frequency at 99th percentile to reduce extreme outlier influence
    X = X.copy()
    X["monetary"] = X["monetary"].clip(upper=X["monetary"].quantile(0.99))
    X["frequency"] = X["frequency"].clip(upper=X["frequency"].quantile(0.99))
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k,random_state=42)


    labels = kmeans.fit_predict(X_scaled)
    rfm["Cluster"] = labels
    sil = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score (k={k}): {sil:.3f}  [range: -1 to 1, higher is better]")
    cluster_means = rfm.groupby("Cluster")[["recency", "frequency", "monetary"]].mean()
    print(cluster_means.round(2))

    # Derive labels from data: lowest recency + highest monetary = High Value,
    # highest recency = Churned, remainder = Active Regular
    high_value = cluster_means["monetary"].idxmax()
    churned = cluster_means["recency"].idxmax()
    active = [c for c in cluster_means.index if c not in (high_value, churned)][0]

    cluster_names = {high_value: "High Value", churned: "Churned", active: "Active Regular"}
    rfm["Cluster_Name"] = rfm["Cluster"].map(cluster_names)
    print(rfm["Cluster_Name"].value_counts())
    rfm.index = rfm.index.astype(str)

    return rfm, X_scaled



def visualize_clusters(rfm, X_scaled):

    # Reduce to 2 dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    colors = ["steelblue", "tomato", "green"]
    cluster_names = rfm["Cluster_Name"].unique()

    for i, name in enumerate(cluster_names):

        mask = rfm["Cluster_Name"] == name


        plt.scatter(
        X_pca[mask, 0],   # PC1 values for this cluster
        X_pca[mask, 1],   # PC2 values for this cluster
        c=colors[i],
        label=name,
        alpha=0.5
        )

    plt.title("Customer Segments — PCA Visualization")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/cluster_plot.png")
    plt.show()


def compute_churn_risk(rfm):
    """
    Estimates churn risk for each customer based on recency.
    Higher recency (hasn't bought in a long time) = higher churn risk.
    """
    min_recency = rfm["recency"].min()
    max_recency = rfm["recency"].max()


    # Higher recency = bought longer ago = higher churn risk
    rfm["Churn_Risk"] = (rfm["recency"] - min_recency) / (max_recency - min_recency)

    print("\n=== Churn Risk by Segment ===")
    print(rfm.groupby("Cluster_Name")["Churn_Risk"].mean().round(2))

    return rfm


if __name__ == "__main__":
    cleaned_data = pd.read_csv("data/cleaned_data.csv",parse_dates=["InvoiceDate"])
    rfm = compute_rfm(cleaned_data)

    #plot_elbow(rfm)
    fit_clusters(rfm,k=3)
    compute_churn_risk(rfm)
    rfm.to_csv("data/rfm_data.csv")
    print("RFM data saved to data/rfm_data.csv")
