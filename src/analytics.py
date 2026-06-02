import pandas as pd
import matplotlib.pyplot as plt

def assign_segement(score):

    if score >= 12:
        return "Champions"
    elif score >= 9:
        return "Loyal"
    elif score>=6:
        return "At Risk"
    else:
        return "Lost"

def compute_rfm(df):

    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        recency = ("InvoiceDate", lambda x: (reference_date-x.max()).days),
        frequency = ("Invoice","nunique"),
        monetary = ("TotalPrice","sum")
    )

    rfm["R-Score"] = pd.qcut(rfm["recency"],q=5,labels=[5,4,3,2,1])
    rfm["F-Score"] = pd.qcut(rfm["frequency"].rank(method = "first"),q=5,labels=[1,2,3,4,5])
    rfm["M-Score"] = pd.qcut(rfm["monetary"].rank(method = "first"),q=5,labels=[1,2,3,4,5])

    rfm["RFM_Score"] = rfm["R-Score"].astype(int)+ rfm["F-Score"].astype(int) + rfm["M-Score"].astype(int)

    rfm["Segment"] = rfm["RFM_Score"].apply(assign_segement)

    print("\n=== RFM Analysis Complete ===")
    print(f"Total customers analysed: {len(rfm)}")
    print(f"\nSegment Distribution:")
    print(rfm["Segment"].value_counts())
    print(f"\nRFM Summary Statistics:")
    print(rfm[["recency", "frequency", "monetary"]].describe().round(2))

    print(f"Total unique customers in cleaned data: {df['CustomerID'].nunique()}")
    print(f"Total customers in RFM table: {len(rfm)}")

    return rfm

def monthly_revenue(df):

    df["Month"] = df["InvoiceDate"].dt.to_period("M")

    mr = df.groupby("Month").agg(
        Total_Monthly_Revenue=("TotalPrice", "sum")
    )


    plt.figure(figsize=(12, 5))
    plt.plot(mr.index.astype(str), mr["Total_Monthly_Revenue"], marker="o")
    plt.title("Monthly Revenue Trend")
    plt.xlabel("Month")
    plt.ylabel("Revenue (£)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("data/monthly_revenue.png")
    plt.show()

    return mr

def purchase_patterns(df):

    df["Hour"] = df["InvoiceDate"].dt.hour
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()

    hourly_counts = df.groupby("Hour")["Invoice"].nunique()

    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    daily_counts = df.groupby("DayOfWeek")["Invoice"].nunique()
    daily_counts = daily_counts.reindex(day_order)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(hourly_counts.index, hourly_counts.values, color="steelblue")
    axes[0].set_title("Orders by Hour of Day")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Number of Orders")

    axes[1].bar(daily_counts.index,daily_counts.values,color="red")
    axes[1].set_title("Daily Orders")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Number Of Orders")

    plt.tight_layout()
    plt.savefig("data/purchase_patterns.png")
    plt.show()

def top_products(df,n=10):

    df = df[~df["Description"].str.upper().isin(["POSTAGE", "MANUAL", "DOTCOM POSTAGE", "CRUK COMMISSION"])]
    top = df.groupby("Description")["TotalPrice"].sum()
    top = top.sort_values(ascending=False).head(n)

    plt.figure(figsize=(10, 5))
    top.plot(kind="barh", color="steelblue")
    plt.title(f"Top {n} Products by Revenue")
    plt.xlabel("Total Revenue (£)")
    plt.tight_layout()
    plt.savefig("data/top_products.png")
    plt.show()

    return top


if __name__=="__main__":
    cleaned_data = pd.read_csv("data/cleaned_data.csv",parse_dates=["InvoiceDate"])
    compute_rfm(cleaned_data)
    monthly_revenue(cleaned_data)
    purchase_patterns(cleaned_data)
    top_products(cleaned_data)





