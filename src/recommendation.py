import pandas as pd

def recommend_popular(df,n=5):
    """Returns top N products by total revenue."""

    df = df[~df["Description"].str.upper().isin(["POSTAGE", "MANUAL", "DOTCOM POSTAGE", "CRUK COMMISSION"])]

    top = df.groupby("Description")["TotalPrice"].sum()
    top = top.sort_values(ascending=False).head(n)

    return top

def recommend_for_customer(customer_id,df,rfm,n=5):
    """Return top N products for customer based on the total revenue of the product bought by the same cluster customers"""

    customer_id = int(customer_id)
    customer_cluster = rfm.loc[customer_id,"Cluster_Name"]
    same_cluster_customers = rfm[rfm["Cluster_Name"]== customer_cluster].index

    cluster_df = df[df["CustomerID"].isin(same_cluster_customers)]
    cluster_df = cluster_df[~cluster_df["Description"].str.upper().isin(["POSTAGE", "MANUAL", "DOTCOM POSTAGE", "CRUK COMMISSION"])]
    top = cluster_df.groupby("Description")["TotalPrice"].sum()
    top = top.sort_values(ascending=False).head(n)

    return top

if __name__ == "__main__":
    cleaned_data = pd.read_csv("data/cleaned_data.csv", parse_dates=["InvoiceDate"])
    rfm = pd.read_csv("data/rfm_data.csv", index_col="CustomerID")

    print("=== Popular Products ===")
    print(recommend_popular(cleaned_data))

    for cluster_name in rfm["Cluster_Name"].unique():
        customer_id = rfm[rfm["Cluster_Name"] == cluster_name].index[0]
        print(f"\n=== {cluster_name} (Customer {customer_id}) ===")
        print(recommend_for_customer(customer_id, cleaned_data, rfm))