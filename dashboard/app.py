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


from recommendation import recommend_for_customer, recommend_popular

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