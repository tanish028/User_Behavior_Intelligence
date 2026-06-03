import sys
import os

# Resolve paths relative to the repo root regardless of where streamlit is launched from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))

import streamlit as st
import pandas as pd

st.title("User Behavior Intelligence System")
st.write("Analyzing customer behavior from Online Retail II dataset")

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(ROOT, "data", "cleaned_data.csv"), parse_dates=["InvoiceDate"])
    rfm = pd.read_csv(os.path.join(ROOT, "data", "rfm_data.csv"), index_col="CustomerID")
    return df, rfm

df, rfm = load_data()

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", f"{rfm.shape[0]:,}")
col2.metric("Total Revenue", f"£{df['TotalPrice'].sum():,.0f}")
col3.metric("Total Orders", f"{df['Invoice'].nunique():,}")

st.header("Customer Segments")

col1, col2 = st.columns(2)

# Segment distribution bar chart
with col1:
    st.subheader("RFM Segments")
    segment_counts = rfm["Segment"].value_counts()
    st.bar_chart(segment_counts)

# Cluster distribution bar chart
with col2:
    st.subheader("KMeans Clusters")
    cluster_counts = rfm["Cluster_Name"].value_counts()
    st.bar_chart(cluster_counts)

st.header("Sales Trends")

# Prepare monthly revenue data
df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
monthly = df.groupby("Month")["TotalPrice"].sum()

st.subheader("Monthly Revenue")
st.line_chart(monthly)


st.header("Geographic Revenue Breakdown")

country_revenue = (
    df[df["Country"] != "Unspecified"]
    .groupby("Country")["TotalPrice"]
    .sum()
    .sort_values(ascending=False)
)
top10_countries = country_revenue.head(10)
rest = country_revenue.iloc[10:].sum()
top10_countries["Other (31 countries)"] = rest

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top 10 Countries by Revenue")
    st.bar_chart(top10_countries)
with col2:
    st.subheader("Revenue Share")
    uk = country_revenue.get("United Kingdom", 0)
    non_uk = country_revenue.sum() - uk
    st.metric("UK Revenue", f"£{uk:,.0f}", f"{uk/country_revenue.sum()*100:.1f}% of total")
    st.metric("International Revenue", f"£{non_uk:,.0f}", f"{non_uk/country_revenue.sum()*100:.1f}% of total")
    st.metric("Countries Served", f"{df['Country'].nunique()}")


from recommendation import recommend_for_customer, recommend_popular
from cohort_analysis import build_cohort_matrix, plot_cohort_heatmap

st.header("Cohort Retention Analysis")
st.write("Shows what % of each monthly cohort returned to purchase in subsequent months.")

@st.cache_data
def get_cohort_data(_df):
    return build_cohort_matrix(_df)

cohort_pct, cohort_size = get_cohort_data(df)

max_periods = st.slider("Months to display", min_value=3, max_value=12, value=12)
fig = plot_cohort_heatmap(cohort_pct, max_periods=max_periods)
st.pyplot(fig)

col1, col2, col3 = st.columns(3)
avg_m1 = cohort_pct[1].mean()
avg_m3 = cohort_pct[3].mean() if 3 in cohort_pct.columns else 0
avg_m6 = cohort_pct[6].mean() if 6 in cohort_pct.columns else 0
col1.metric("Avg Month 1 Retention", f"{avg_m1:.1f}%")
col2.metric("Avg Month 3 Retention", f"{avg_m3:.1f}%")
col3.metric("Avg Month 6 Retention", f"{avg_m6:.1f}%")

st.header("Product Recommendations")

customer_ids = rfm.index.tolist()
selected_customer = st.selectbox("Select a Customer ID", customer_ids)

if selected_customer:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Popular Products")
        popular = recommend_popular(df)
        st.dataframe(popular.reset_index())

    with col2:
        st.subheader(f"Recommended for Customer {selected_customer}")
        recommended = recommend_for_customer(selected_customer, df, rfm)
        st.dataframe(recommended.reset_index())



st.sidebar.title("About")
st.sidebar.write("**Dataset:** Online Retail II (UCI)")
st.sidebar.write("**Customers:** 5,878")
st.sidebar.write("**Period:** 2009-2011")
st.sidebar.write("**Built with:** Python, pandas, sklearn, streamlit")

st.sidebar.title("Segments")
st.sidebar.write("🏆 Champions — Recent, frequent, high spend")
st.sidebar.write("✅ Active Regular — Moderate activity")
st.sidebar.write("⚠️ Churned — Long inactive, low spend")
st.sidebar.write("💎 High Value — Wholesalers, very high spend")