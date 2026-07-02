import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))

import streamlit as st
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="User Behavior Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("📊 User Behavior Intelligence")
st.caption("Customer analytics on the UCI Online Retail II dataset — 1M+ transactions, 5,878 customers, 2009–2011")

# ── Data loading ───────────────────────────────────────────────────────────────
from recommendation import recommend_for_customer, recommend_popular
from cohort_analysis import build_cohort_matrix, plot_cohort_heatmap
from churn_model import (
    create_churn_label, train_churn_model, evaluate_model,
    plot_feature_importance, predict_churn,
    compute_shap_values, plot_shap_summary, plot_shap_waterfall
)
from markov_analysis import (
    compute_monthly_segments, build_transition_matrix,
    plot_transition_heatmap, compute_steady_state
)

@st.cache_data
def load_data():
    df  = pd.read_csv(os.path.join(ROOT, "data", "cleaned_data.csv"), parse_dates=["InvoiceDate"])
    rfm = pd.read_csv(os.path.join(ROOT, "data", "rfm_data.csv"), index_col="CustomerID")
    return df, rfm

@st.cache_data
def get_cohort_data(_df):
    return build_cohort_matrix(_df)

@st.cache_data
def get_markov_data(_df):
    monthly_segments = compute_monthly_segments(_df)
    transition_matrix = build_transition_matrix(monthly_segments)
    steady = compute_steady_state(transition_matrix)
    return transition_matrix, steady

@st.cache_resource
def train_model(_rfm, _version=3):
    rfm_labeled = create_churn_label(_rfm, recency_threshold=180)
    model, X_test, y_test, features = train_churn_model(rfm_labeled)
    return model, X_test, y_test, features, rfm_labeled

df, rfm = load_data()

# ── Top-level KPIs (always visible) ───────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Customers",  f"{rfm.shape[0]:,}")
k2.metric("Total Revenue",    f"£{df['TotalPrice'].sum():,.0f}")
k3.metric("Total Orders",     f"{df['Invoice'].nunique():,}")
k4.metric("Countries Served", f"{df['Country'].nunique()}")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Overview",
    "🌍 Geographic",
    "👥 Cohort Retention",
    "🔄 Customer Journeys",
    "🤖 Churn Prediction",
    "🛍️ Recommendations",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Customer Segments")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RFM Segments")
        segment_counts = rfm["Segment"].value_counts()
        st.bar_chart(segment_counts)
    with col2:
        st.subheader("K-Means Clusters")
        cluster_counts = rfm["Cluster_Name"].value_counts()
        st.bar_chart(cluster_counts)

    st.info(
        "**💡 Insight:** The majority of customers fall into 'At Risk' or 'Lost' — "
        "this is normal for e-commerce datasets where one-time buyers are common. "
        "The High Value cluster likely represents wholesale/B2B buyers who buy rarely "
        "but in very large quantities."
    )

    st.divider()
    st.header("Sales Trends")

    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    monthly = df.groupby("Month")["TotalPrice"].sum()
    st.subheader("Monthly Revenue")
    st.line_chart(monthly)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Revenue by Day of Week")
        df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        day_rev = df.groupby("DayOfWeek")["TotalPrice"].sum().reindex(day_order)
        st.bar_chart(day_rev)
    with col2:
        st.subheader("Revenue by Hour of Day")
        df["Hour"] = df["InvoiceDate"].dt.hour
        hour_rev = df.groupby("Hour")["TotalPrice"].sum()
        st.bar_chart(hour_rev)

    st.info(
        "**💡 Insight:** Revenue spikes sharply in **October–November** each year — "
        "classic seasonal retail (gift buying). Orders peak between **10 AM and 3 PM on weekdays** "
        "and drop to near-zero on Sundays. This ordering pattern is more typical of a "
        "**B2B wholesale** customer base than individual retail consumers."
    )
    st.warning(
        "**⚠️ Data quality note:** ~24% of raw transactions had no Customer ID and were "
        "dropped before RFM analysis. In a production system, these would be worth "
        "recovering via session-based tracking."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GEOGRAPHIC
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Geographic Revenue Breakdown")

    country_revenue = (
        df[df["Country"] != "Unspecified"]
        .groupby("Country")["TotalPrice"]
        .sum()
        .sort_values(ascending=False)
    )
    top10 = country_revenue.head(10)
    rest  = country_revenue.iloc[10:].sum()
    top10_with_rest = top10.copy()
    top10_with_rest["Other (31 countries)"] = rest

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Top 10 Countries by Revenue")
        st.bar_chart(top10_with_rest)
    with col2:
        st.subheader("Revenue Split")
        uk     = country_revenue.get("United Kingdom", 0)
        non_uk = country_revenue.sum() - uk
        st.metric("UK Revenue",            f"£{uk:,.0f}",     f"{uk/country_revenue.sum()*100:.1f}% of total")
        st.metric("International Revenue", f"£{non_uk:,.0f}", f"{non_uk/country_revenue.sum()*100:.1f}% of total")

    st.info(
        "**💡 Insight:** The UK accounts for ~85% of total revenue — heavy geographic concentration. "
        "The Netherlands, Ireland, Germany, and France are the largest international markets. "
        "Expanding retention campaigns to these four countries could meaningfully grow "
        "international revenue without acquiring new markets."
    )
    st.warning(
        "**⚠️ Concentration risk:** With 85% of revenue tied to one country, any UK-specific "
        "disruption (economic, regulatory, or logistical) has outsized business impact."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COHORT RETENTION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Cohort Retention Analysis")
    st.write(
        "Groups customers by the month they first purchased, then tracks what percentage "
        "came back in each subsequent month. Each row is one cohort; each column is months since first purchase."
    )

    cohort_pct, cohort_size = get_cohort_data(df)

    max_periods = st.slider("Months to display", min_value=3, max_value=12, value=12)
    fig = plot_cohort_heatmap(cohort_pct, max_periods=max_periods)
    st.pyplot(fig)

    col1, col2, col3 = st.columns(3)
    avg_m1 = cohort_pct[1].mean() if 1 in cohort_pct.columns else 0
    avg_m3 = cohort_pct[3].mean() if 3 in cohort_pct.columns else 0
    avg_m6 = cohort_pct[6].mean() if 6 in cohort_pct.columns else 0
    col1.metric("Avg Month-1 Retention", f"{avg_m1:.1f}%")
    col2.metric("Avg Month-3 Retention", f"{avg_m3:.1f}%")
    col3.metric("Avg Month-6 Retention", f"{avg_m6:.1f}%")

    st.error(
        f"**🚨 Key finding:** Only **{avg_m1:.1f}%** of new customers make a second purchase within "
        "the first month — meaning **3 in 4 new customers are lost immediately**. "
        "This is the single biggest lever for revenue growth: a 10-point improvement in "
        "Month-1 retention would be worth more than acquiring 10% more new customers."
    )
    st.info(
        "**💡 Recommended action:** Trigger an automated follow-up email or discount offer "
        "within 7 days of a customer's first purchase. Customers who return once are "
        "significantly more likely to become regulars (see Month-3 and Month-6 rates, "
        "which are more stable once customers have re-purchased)."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CUSTOMER JOURNEYS (MARKOV)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Customer Segment Transitions — Markov Chain")
    st.write(
        "Each month, every customer is assigned an RFM segment. This model tracks how "
        "customers move between segments and predicts where the customer base will settle long-term."
    )

    with st.spinner("Computing monthly segments and transition matrix..."):
        transition_matrix, steady = get_markov_data(df)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Transition Probability Heatmap")
        markov_fig = plot_transition_heatmap(transition_matrix)
        st.pyplot(markov_fig)
        st.caption(
            "Each cell = probability of moving from the row segment (this month) "
            "to the column segment (next month). Diagonal = staying in same segment."
        )
    with col2:
        st.subheader("Long-Run Steady State")
        st.write("Where customers settle if current trends continue:")
        for segment, pct in steady.items():
            st.metric(segment, f"{pct:.1f}%")

    st.error(
        f"**🚨 Key finding:** The steady-state model projects that **{steady.get('Lost', 0):.1f}%** "
        "of the customer base will eventually end up in the 'Lost' segment if no intervention occurs. "
        "This is the long-run churn floor — the baseline you're fighting against."
    )
    st.info(
        f"**💡 Insight:** Champions have a **{steady.get('Champions', 0):.1f}%** steady-state share, "
        "meaning strong customers do exist and persist. The priority should be identifying "
        "customers who are transitioning from 'Loyal' → 'At Risk' and intervening before "
        "they reach 'Lost' — where recovery becomes very expensive."
    )
    st.info(
        "**💡 How to read the heatmap:** A high value on the diagonal means customers tend "
        "to stay in that segment (sticky). A high off-diagonal value (e.g., Loyal → At Risk) "
        "is a warning signal — customers are sliding down faster than expected."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CHURN PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Churn Prediction Model — XGBoost")
    st.write(
        "A customer is defined as **churned** if they haven't purchased in 180+ days. "
        "The model predicts this using only purchase frequency (F-Score) and spend (M-Score) — "
        "recency is excluded to avoid data leakage."
    )

    model, X_test, y_test, features, rfm_labeled = train_model(rfm)
    auc, eval_fig = evaluate_model(model, X_test, y_test)

    col1, col2, col3 = st.columns(3)
    col1.metric("ROC-AUC", f"{auc:.3f}", help="0.5 = random | 0.75+ = solid | 0.85+ = very good")
    col2.metric("Churned Customers", f"{rfm_labeled['Churned'].sum():,}")
    col3.metric("Churn Rate", f"{rfm_labeled['Churned'].mean()*100:.1f}%")

    st.info(
        f"**💡 What ROC-AUC {auc:.3f} means:** If you randomly pick one churner and one active customer, "
        f"the model correctly ranks the churner as riskier **{auc*100:.1f}% of the time**. "
        "Random guessing would score 50%. This model is meaningfully better than guessing, "
        "despite using only 2 features."
    )

    with st.expander("📈 Model Evaluation — Confusion Matrix & ROC Curve"):
        st.pyplot(eval_fig)
        st.caption(
            "Confusion matrix: how many churners vs active customers the model correctly classified. "
            "ROC curve: the trade-off between catching real churners (recall) vs false alarms (precision)."
        )

    with st.expander("🔍 Feature Importance — What actually drives churn?"):
        imp_fig = plot_feature_importance(model, features)
        st.pyplot(imp_fig)
        st.info(
            "**💡 Insight:** F-Score (purchase frequency) accounts for ~91% of the model's predictive power. "
            "**How often a customer buys matters far more than how much they spend.** "
            "A customer who buys small amounts 5× a month is much harder to lose than "
            "one who made a single large purchase — even if the total spend is the same."
        )

    with st.expander("🧠 SHAP Explainability — Why did the model flag this customer?"):
        st.write(
            "SHAP (SHapley Additive exPlanations) goes beyond overall feature importance — "
            "it explains each individual prediction. For every customer, it shows exactly "
            "how much each feature pushed their churn probability up or down."
        )

        with st.spinner("Computing SHAP values..."):
            explainer, shap_values = compute_shap_values(model, X_test)

        st.markdown("**Average feature impact across all customers:**")
        shap_summary_fig = plot_shap_summary(shap_values, X_test)
        st.pyplot(shap_summary_fig)
        st.caption(
            "Bar length = how much that feature moves the churn probability on average. "
            "F-Score dominates — consistent with the feature importance chart above."
        )

        st.markdown("**Drill into a specific customer:**")
        max_idx = len(X_test) - 1
        customer_idx = st.slider("Customer index (test set)", 0, min(max_idx, 50), 0)
        shap_wf_fig = plot_shap_waterfall(explainer, shap_values, X_test, customer_idx=customer_idx)
        st.pyplot(shap_wf_fig)
        st.caption(
            "🔴 Red = feature increased churn risk  |  🔵 Blue = feature reduced churn risk. "
            "The chart starts at the average churn probability and adds each feature's contribution "
            "to reach this customer's final prediction."
        )

    st.divider()
    st.subheader("🎯 Live Churn Predictor")
    st.write("Enter any customer's scores to instantly estimate their churn risk.")
    st.caption(
        "R-Score is excluded as a feature — recency directly defines the churn label, "
        "so including it would be data leakage (the model would just re-learn the rule used to create the label)."
    )

    col1, col2 = st.columns(2)
    with col1:
        f = st.slider("F-Score (Frequency)",  1, 5, 3, help="5 = very frequent buyer | 1 = rarely buys")
    with col2:
        m = st.slider("M-Score (Monetary)", 1, 5, 3, help="5 = very high spend | 1 = low spend")

    prob = predict_churn(model, f, m)
    pct  = prob * 100

    if pct >= 70:
        st.error(f"⚠️ Churn Probability: **{pct:.1f}%** — High Risk. Prioritise for re-engagement campaign.")
    elif pct >= 40:
        st.warning(f"🔶 Churn Probability: **{pct:.1f}%** — Medium Risk. Monitor and consider a targeted offer.")
    else:
        st.success(f"✅ Churn Probability: **{pct:.1f}%** — Low Risk. Customer shows healthy engagement signals.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("Cluster-Aware Product Recommendations")
    st.write(
        "Recommendations are tailored to the customer's K-Means cluster — "
        "a High Value wholesale buyer gets different suggestions than an Active Regular. "
        "Popular products are shown as a baseline for comparison."
    )

    customer_ids = rfm.index.tolist()
    selected_customer = st.selectbox("Select a Customer ID", customer_ids)

    if selected_customer:
        cluster = rfm.loc[selected_customer, "Cluster_Name"] if "Cluster_Name" in rfm.columns else "Unknown"
        segment = rfm.loc[selected_customer, "Segment"]     if "Segment"      in rfm.columns else "Unknown"

        col1, col2, col3 = st.columns(3)
        col1.metric("Customer Segment", segment)
        col2.metric("K-Means Cluster",  cluster)
        churn_prob = predict_churn(model, int(rfm.loc[selected_customer, "F-Score"]), int(rfm.loc[selected_customer, "M-Score"]))
        col3.metric("Churn Risk", f"{churn_prob*100:.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏆 Popular Products (all customers)")
            popular = recommend_popular(df)
            st.dataframe(popular.reset_index(), use_container_width=True)
        with col2:
            st.subheader(f"✨ Recommended for Customer {selected_customer}")
            recommended = recommend_for_customer(selected_customer, df, rfm)
            st.dataframe(recommended.reset_index(), use_container_width=True)

        st.info(
            f"**💡 Insight:** This customer is in the **{cluster}** cluster and the **{segment}** RFM segment. "
            "Cluster-aware recommendations filter to products popular within this customer's peer group, "
            "rather than globally popular items — increasing relevance and purchase likelihood."
        )


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("About")
st.sidebar.write("**Dataset:** UCI Online Retail II")
st.sidebar.write("**Customers:** 5,878")
st.sidebar.write("**Transactions:** ~1M rows")
st.sidebar.write("**Period:** Dec 2009 – Dec 2011")
st.sidebar.divider()
st.sidebar.title("RFM Segments")
st.sidebar.write("🏆 **Champions** — Recent, frequent, high spend")
st.sidebar.write("✅ **Loyal** — Regular buyers, solid spend")
st.sidebar.write("⚠️ **At Risk** — Haven't bought recently")
st.sidebar.write("💀 **Lost** — Long inactive, low engagement")
st.sidebar.divider()
st.sidebar.title("Model")
st.sidebar.write("**Algorithm:** XGBoost Classifier")
st.sidebar.write("**ROC-AUC:** 0.778")
st.sidebar.write("**Churn definition:** No purchase in 180+ days")
st.sidebar.write("**Explainability:** SHAP TreeExplainer")
