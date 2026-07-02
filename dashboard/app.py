import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="User Behavior Intelligence",
    page_icon="📊",
    layout="wide",
)

# ── CSS: animations, card styles, chart consistency ───────────────────────────
st.markdown("""
<style>
/* Fade-in for the whole page on load */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.main .block-container {
    animation: fadeInUp 0.45s ease-out;
    padding-top: 1.5rem;
}

/* KPI metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
    border: 1px solid #DBEAFE;
    border-radius: 0.75rem;
    padding: 1rem 1.2rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.12);
}

/* Tab bar */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #F1F5F9;
    border-radius: 0.6rem;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0.5rem;
    padding: 0.4rem 1rem;
    transition: background 0.2s ease, color 0.2s ease;
}

/* Expander hover */
.streamlit-expanderHeader {
    transition: background-color 0.2s ease;
    border-radius: 0.5rem;
}
.streamlit-expanderHeader:hover {
    background-color: #EFF6FF !important;
}

/* Key findings card grid */
.kf-grid {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
}
.kf-card {
    flex: 1;
    min-width: 160px;
    background: #fff;
    border-radius: 0.75rem;
    padding: 1rem 1.2rem;
    border-left: 4px solid;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    animation: fadeInUp 0.5s ease-out both;
}
.kf-card.blue  { border-color: #2563EB; }
.kf-card.green { border-color: #10B981; }
.kf-card.amber { border-color: #F59E0B; }
.kf-card.red   { border-color: #EF4444; }
.kf-card h2 { margin: 0 0 4px; font-size: 1.7rem; font-weight: 700; }
.kf-card p  { margin: 0; font-size: 0.82rem; color: #64748B; line-height: 1.4; }
</style>
""", unsafe_allow_html=True)

# ── Consistent matplotlib style ────────────────────────────────────────────────
PALETTE  = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
            "#06B6D4", "#EC4899", "#84CC16", "#F97316", "#6366F1"]
BG       = "#F8FAFC"
BORDER   = "#334155"

def _style_ax(ax, title="", xlabel="", ylabel="", grid_axis="x"):
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    ax.tick_params(colors="#475569", labelsize=9)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color=BORDER, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color="#475569")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color="#475569")
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E2E8F0", linewidth=0.7, linestyle="--")
        ax.set_axisbelow(True)

def _fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    return fig, ax

# ── Data / model loading ───────────────────────────────────────────────────────
from recommendation import recommend_for_customer, recommend_popular, evaluate_recommendations
from cohort_analysis import build_cohort_matrix, plot_cohort_heatmap
from churn_model import (
    create_churn_label, train_churn_model, evaluate_model,
    plot_feature_importance, predict_churn,
    compute_shap_values, plot_shap_summary, plot_shap_waterfall,
)
from markov_analysis import (
    compute_monthly_segments, build_transition_matrix,
    plot_transition_heatmap, compute_steady_state,
)
from architecture_diagram import generate_diagram

@st.cache_data
def load_data():
    df  = pd.read_csv(os.path.join(ROOT, "data", "cleaned_data.csv"),
                      parse_dates=["InvoiceDate"])
    rfm = pd.read_csv(os.path.join(ROOT, "data", "rfm_data.csv"),
                      index_col="CustomerID")
    return df, rfm

@st.cache_data
def get_cohort_data(_df):
    return build_cohort_matrix(_df)

@st.cache_data
def get_markov_data(_df):
    ms  = compute_monthly_segments(_df)
    tm  = build_transition_matrix(ms)
    ss  = compute_steady_state(tm)
    return tm, ss

@st.cache_resource
def train_model(_rfm, _version=3):
    rfm_l = create_churn_label(_rfm, recency_threshold=180)
    model, X_test, y_test, features = train_churn_model(rfm_l)
    return model, X_test, y_test, features, rfm_l

@st.cache_data
def get_rec_metrics(_df, _rfm):
    return evaluate_recommendations(_df, _rfm, k=10, sample_size=200)

df, rfm = load_data()
df["Month"]     = df["InvoiceDate"].dt.to_period("M").astype(str)
df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
df["Hour"]      = df["InvoiceDate"].dt.hour

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📊 User Behavior Intelligence")
st.caption("End-to-end customer analytics · UCI Online Retail II · 1M+ transactions · 5,878 customers · 2009–2011")

# ── Top KPIs (always visible) ──────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers",       f"{rfm.shape[0]:,}")
k2.metric("Total Revenue",   f"£{df['TotalPrice'].sum():,.0f}")
k3.metric("Total Orders",    f"{df['Invoice'].nunique():,}")
k4.metric("Countries",       f"{df['Country'].nunique()}")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Overview",
    "🌍 Geographic",
    "👥 Cohort Retention",
    "🔄 Customer Journeys",
    "🤖 Churn Prediction",
    "🛍️ Recommendations",
    "🗺️ Architecture",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # Key Findings cards
    churn_pct = round((rfm["Segment"].isin(["At Risk","Lost"])).mean() * 100, 1)
    monthly   = df.groupby("Month")["TotalPrice"].sum()
    peak_m    = monthly.idxmax()

    st.markdown(f"""
    <div class="kf-grid">
      <div class="kf-card blue">
        <h2>{rfm.shape[0]:,}</h2>
        <p>Unique customers across<br>41 countries, 2009–2011</p>
      </div>
      <div class="kf-card red">
        <h2>{churn_pct}%</h2>
        <p>Customers in At-Risk or<br>Lost segments right now</p>
      </div>
      <div class="kf-card green">
        <h2>£{df['TotalPrice'].sum()/1e6:.1f}M</h2>
        <p>Total revenue generated<br>across the full period</p>
      </div>
      <div class="kf-card amber">
        <h2>{peak_m}</h2>
        <p>Peak revenue month —<br>Q4 seasonal spike every year</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Segment charts ──
    st.subheader("Customer Segments")
    c1, c2 = st.columns(2)

    with c1:
        seg_counts = rfm["Segment"].value_counts()
        fig, ax = _fig(5, 3.5)
        bars = ax.barh(seg_counts.index, seg_counts.values,
                       color=["#2563EB","#10B981","#F59E0B","#EF4444"][:len(seg_counts)],
                       edgecolor="white", height=0.6)
        for bar, v in zip(bars, seg_counts.values):
            ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height()/2,
                    f"{v:,}", va="center", fontsize=9, color=BORDER)
        _style_ax(ax, title="RFM Segments", xlabel="Customers", grid_axis="x")
        ax.set_xlim(0, seg_counts.max() * 1.18)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        cl_counts = rfm["Cluster_Name"].value_counts()
        fig, ax = _fig(5, 3.5)
        bars = ax.barh(cl_counts.index, cl_counts.values,
                       color=["#8B5CF6","#06B6D4","#EC4899"][:len(cl_counts)],
                       edgecolor="white", height=0.6)
        for bar, v in zip(bars, cl_counts.values):
            ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height()/2,
                    f"{v:,}", va="center", fontsize=9, color=BORDER)
        _style_ax(ax, title="K-Means Clusters", xlabel="Customers", grid_axis="x")
        ax.set_xlim(0, cl_counts.max() * 1.18)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        "**💡 Insight:** Most customers fall into 'At Risk' or 'Lost' — normal for e-commerce "
        "with high one-time buyer rates. The High Value cluster likely represents wholesale/B2B "
        "buyers who buy rarely but in very large quantities."
    )

    st.divider()
    st.subheader("Sales Trends")

    # Monthly revenue
    fig, ax = _fig(11, 3.8)
    x_vals = range(len(monthly))
    ax.fill_between(x_vals, monthly.values, alpha=0.15, color="#2563EB")
    ax.plot(x_vals, monthly.values, color="#2563EB", lw=2, marker="o",
            markersize=3.5, markerfacecolor="white", markeredgewidth=1.2)
    step = max(1, len(monthly) // 10)
    ax.set_xticks(list(x_vals)[::step])
    ax.set_xticklabels(list(monthly.index)[::step], rotation=35, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v/1000:.0f}K"))
    _style_ax(ax, title="Monthly Revenue", ylabel="Revenue", grid_axis="y")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    c1, c2 = st.columns(2)
    with c1:
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        day_rev   = df.groupby("DayOfWeek")["TotalPrice"].sum().reindex(day_order)
        fig, ax   = _fig(5, 3.5)
        colors    = ["#2563EB" if d != "Sunday" else "#EF4444" for d in day_order]
        ax.bar(day_order, day_rev.values, color=colors, edgecolor="white", width=0.6)
        ax.set_xticklabels([d[:3] for d in day_order], fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v/1000:.0f}K"))
        _style_ax(ax, title="Revenue by Day of Week", ylabel="Revenue", grid_axis="y")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        hour_rev = df.groupby("Hour")["TotalPrice"].sum()
        fig, ax  = _fig(5, 3.5)
        peak_hrs = hour_rev.nlargest(3).index
        bar_colors = ["#F59E0B" if h in peak_hrs else "#2563EB" for h in hour_rev.index]
        ax.bar(hour_rev.index, hour_rev.values, color=bar_colors, edgecolor="white", width=0.75)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v/1000:.0f}K"))
        ax.set_xlabel("Hour of day (24h)", fontsize=9, color="#475569")
        _style_ax(ax, title="Revenue by Hour of Day (amber = peak)", ylabel="Revenue", grid_axis="y")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        "**💡 Insight:** Revenue spikes in **October–November** each year (seasonal gift buying). "
        "Orders peak **10 AM–3 PM on weekdays** and drop to near-zero on Sundays — "
        "a pattern typical of a **B2B wholesale** customer base."
    )
    st.warning(
        "**⚠️ Data quality:** ~24% of raw transactions had no Customer ID and were excluded "
        "from RFM analysis. In production, session-based tracking would recover these."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GEOGRAPHIC
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Geographic Revenue Breakdown")

    country_revenue = (
        df[df["Country"] != "Unspecified"]
        .groupby("Country")["TotalPrice"].sum()
        .sort_values(ascending=False)
    )
    uk     = country_revenue.get("United Kingdom", 0)
    non_uk = country_revenue.sum() - uk
    top10  = country_revenue.head(10)

    c1, c2 = st.columns([2, 1])
    with c1:
        fig, ax = _fig(8, 4.5)
        bars = ax.barh(top10.index[::-1], top10.values[::-1],
                       color=["#EF4444" if c == "United Kingdom" else "#2563EB"
                              for c in top10.index[::-1]],
                       edgecolor="white", height=0.65)
        for bar, v in zip(bars, top10.values[::-1]):
            ax.text(bar.get_width() + top10.max()*0.01,
                    bar.get_y() + bar.get_height()/2,
                    f"£{v/1000:.0f}K", va="center", fontsize=8.5, color=BORDER)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v/1e6:.1f}M"))
        _style_ax(ax, title="Top 10 Countries by Revenue (red = UK)", grid_axis="x")
        ax.set_xlim(0, top10.max() * 1.18)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.metric("UK Revenue",            f"£{uk:,.0f}",     f"{uk/country_revenue.sum()*100:.1f}% of total")
        st.metric("International Revenue", f"£{non_uk:,.0f}", f"{non_uk/country_revenue.sum()*100:.1f}% of total")
        st.metric("Countries Served",      str(df["Country"].nunique()))

        # Mini pie-style metric bar
        fig2, ax2 = _fig(3.5, 2.5)
        ax2.barh(["International", "United Kingdom"],
                 [non_uk/country_revenue.sum()*100, uk/country_revenue.sum()*100],
                 color=["#2563EB", "#EF4444"], edgecolor="white", height=0.5)
        ax2.set_xlim(0, 100)
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        _style_ax(ax2, title="Revenue Share", grid_axis="x")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.info(
        "**💡 Insight:** UK accounts for ~85% of total revenue — significant concentration risk. "
        "Netherlands, Ireland, Germany, and France are the top international markets. "
        "Targeted retention campaigns in these four countries could grow international "
        "revenue without entering new markets."
    )
    st.warning(
        "**⚠️ Concentration risk:** 85% of revenue tied to one country means any UK-specific "
        "disruption (economic, regulatory, logistical) has outsized business impact."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COHORT RETENTION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Cohort Retention Analysis")
    st.write(
        "Groups customers by their **first purchase month**, then tracks what % returned "
        "in each subsequent month. Each row is a cohort; each column is months since first purchase."
    )

    cohort_pct, _ = get_cohort_data(df)

    max_periods = st.slider("Months to display", 3, 12, 12)
    fig = plot_cohort_heatmap(cohort_pct, max_periods=max_periods)
    st.pyplot(fig)
    plt.close(fig)

    avg_m1 = cohort_pct[1].mean() if 1 in cohort_pct.columns else 0
    avg_m3 = cohort_pct[3].mean() if 3 in cohort_pct.columns else 0
    avg_m6 = cohort_pct[6].mean() if 6 in cohort_pct.columns else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Month-1 Retention", f"{avg_m1:.1f}%")
    c2.metric("Avg Month-3 Retention", f"{avg_m3:.1f}%")
    c3.metric("Avg Month-6 Retention", f"{avg_m6:.1f}%")

    st.error(
        f"**🚨 Key finding:** Only **{avg_m1:.1f}%** of new customers make a second purchase "
        "in month 1 — **3 in 4 are lost immediately**. A 10-point improvement in Month-1 "
        "retention is worth more than acquiring 10% more new customers."
    )
    st.info(
        "**💡 Action:** Trigger an automated follow-up email or discount within 7 days of a "
        "customer's first purchase. Customers who return once become significantly more loyal "
        "(Month-3 and Month-6 rates are much more stable once someone has re-purchased)."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CUSTOMER JOURNEYS (MARKOV)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Customer Segment Transitions — Markov Chain")
    st.write(
        "Every month each customer is re-scored and assigned an RFM segment. "
        "This model tracks how customers **move between segments** and projects "
        "where the customer base will eventually settle."
    )

    with st.spinner("Computing transition matrix..."):
        transition_matrix, steady = get_markov_data(df)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Transition Probability Heatmap")
        mfig = plot_transition_heatmap(transition_matrix)
        st.pyplot(mfig)
        plt.close(mfig)
        st.caption(
            "Each cell = probability of moving from the row segment (this month) "
            "to the column segment (next month). Diagonal = staying in same segment."
        )
    with c2:
        st.subheader("Long-Run Steady State")
        st.write("Where customers settle if current trends continue:")
        for seg, pct in steady.items():
            st.metric(seg, f"{pct:.1f}%")

    st.error(
        f"**🚨 Key finding:** The model projects **{steady.get('Lost', 0):.1f}%** of the customer "
        "base will eventually reach 'Lost' if no intervention occurs. This is the long-run "
        "churn floor you're fighting against."
    )
    st.info(
        f"**💡 Insight:** Champions stabilise at **{steady.get('Champions', 0):.1f}%** long-run. "
        "The priority should be catching customers transitioning Loyal → At Risk **before** they "
        "reach Lost — recovery from Lost is extremely expensive."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CHURN PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Churn Prediction — XGBoost")
    st.write(
        "**Churned** = no purchase in 180+ days. The model predicts this using F-Score "
        "(frequency) and M-Score (spend) only — R-Score is excluded to avoid data leakage."
    )

    model, X_test, y_test, features, rfm_labeled = train_model(rfm)
    auc, eval_fig = evaluate_model(model, X_test, y_test)

    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC",          f"{auc:.3f}",  help="0.5 = random | 0.75+ = solid")
    c2.metric("Churned Customers", f"{rfm_labeled['Churned'].sum():,}")
    c3.metric("Churn Rate",        f"{rfm_labeled['Churned'].mean()*100:.1f}%")

    st.info(
        f"**💡 What AUC {auc:.3f} means:** Pick one random churner and one random active customer — "
        f"the model correctly ranks the churner as riskier **{auc*100:.1f}% of the time**. "
        "Random guessing = 50%."
    )

    with st.expander("📈 Model Evaluation — Confusion Matrix & ROC Curve"):
        st.pyplot(eval_fig)
        plt.close(eval_fig)

    with st.expander("🔍 Feature Importance — What drives churn?"):
        imp_fig = plot_feature_importance(model, features)
        st.pyplot(imp_fig)
        plt.close(imp_fig)
        st.info(
            "**💡 Insight:** F-Score (frequency) accounts for ~91% of predictive power. "
            "**How often a customer buys matters far more than how much they spend.** "
            "A customer buying small amounts 5× a month is far less likely to churn than "
            "one who made a single large purchase — even if the total spend is the same."
        )

    with st.expander("🧠 SHAP Explainability — Why was this customer flagged?"):
        st.write(
            "SHAP explains each individual prediction — not just which features matter "
            "overall, but exactly how much each feature pushed **this customer's** probability up or down."
        )
        with st.spinner("Computing SHAP values..."):
            explainer, shap_values = compute_shap_values(model, X_test)

        st.markdown("**Average feature impact across all customers:**")
        shap_summary_fig = plot_shap_summary(shap_values, X_test)
        st.pyplot(shap_summary_fig)
        plt.close(shap_summary_fig)
        st.caption("Bar length = average shift in churn probability caused by that feature.")

        st.markdown("**Drill into one customer:**")
        max_idx = len(X_test) - 1
        cidx = st.slider("Customer index (test set)", 0, min(max_idx, 50), 0)
        wf_fig = plot_shap_waterfall(explainer, shap_values, X_test, customer_idx=cidx)
        st.pyplot(wf_fig)
        plt.close(wf_fig)
        st.caption(
            "🔴 Red = feature increased churn risk  |  🔵 Blue = feature reduced churn risk. "
            "Starts at the average churn probability and adds each feature's contribution."
        )

    st.divider()
    st.subheader("🎯 Live Churn Predictor")
    st.caption(
        "R-Score excluded — recency directly defines the churn label, "
        "so including it would leak the answer into the model."
    )
    c1, c2 = st.columns(2)
    with c1:
        f = st.slider("F-Score (Frequency)",  1, 5, 3, help="5 = very frequent | 1 = rarely buys")
    with c2:
        m = st.slider("M-Score (Monetary)", 1, 5, 3, help="5 = high spend | 1 = low spend")

    prob = predict_churn(model, f, m)
    pct  = prob * 100
    if pct >= 70:
        st.error(f"⚠️ **{pct:.1f}% churn risk** — High. Prioritise for re-engagement campaign immediately.")
    elif pct >= 40:
        st.warning(f"🔶 **{pct:.1f}% churn risk** — Medium. Monitor and consider a targeted offer.")
    else:
        st.success(f"✅ **{pct:.1f}% churn risk** — Low. Customer shows healthy engagement signals.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("Cluster-Aware Product Recommendations")
    st.write(
        "Recommendations are tailored to the customer's K-Means cluster — "
        "a High Value wholesale buyer gets different suggestions than an Active Regular. "
        "Quality is measured with **Precision@10** and **Recall@10**."
    )

    # Metrics
    with st.spinner("Evaluating recommendation quality (200-customer sample)..."):
        metrics = get_rec_metrics(df, rfm)

    c1, c2, c3 = st.columns(3)
    k = 10
    p_val = metrics.get(f"Precision@{k}", 0)
    r_val = metrics.get(f"Recall@{k}", 0)
    c1.metric(f"Precision@{k}",  f"{p_val:.3f}", help="Of the 10 recommendations, what fraction did the customer actually buy?")
    c2.metric(f"Recall@{k}",     f"{r_val:.3f}", help="Of all items the customer bought, what fraction were recommended?")
    c3.metric("Customers Evaluated", f"{metrics.get('n_evaluated', 0):,}")

    st.info(
        f"**💡 How to read this:** Precision@10 = {p_val:.3f} means on average "
        f"**{p_val*10:.1f} of the 10 recommended items** were things the customer actually bought "
        f"in their next visit. Recall@10 = {r_val:.3f} means the recommendations captured "
        f"**{r_val*100:.1f}% of the items** the customer wanted. "
        "These are evaluated using leave-one-out: the customer's last invoice is held out "
        "as the test set, and recommendations are generated from their remaining history."
    )

    st.divider()
    customer_ids     = rfm.index.tolist()
    selected_customer = st.selectbox("Select a Customer ID", customer_ids)

    if selected_customer:
        cluster    = rfm.loc[selected_customer, "Cluster_Name"] if "Cluster_Name" in rfm.columns else "Unknown"
        segment    = rfm.loc[selected_customer, "Segment"]      if "Segment"      in rfm.columns else "Unknown"
        f_sc       = int(rfm.loc[selected_customer, "F-Score"])
        m_sc       = int(rfm.loc[selected_customer, "M-Score"])
        churn_prob = predict_churn(model, f_sc, m_sc)

        c1, c2, c3 = st.columns(3)
        c1.metric("RFM Segment",   segment)
        c2.metric("K-Means Cluster", cluster)
        c3.metric("Churn Risk",    f"{churn_prob*100:.1f}%")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Global Popular Products")
            st.dataframe(recommend_popular(df).reset_index(), use_container_width=True)
        with c2:
            st.subheader(f"Recommended for Customer {selected_customer}")
            st.dataframe(recommend_for_customer(selected_customer, df, rfm).reset_index(),
                         use_container_width=True)

        st.info(
            f"**💡 Insight:** This customer is in the **{cluster}** cluster ({segment} segment). "
            "Cluster-aware recommendations filter to products popular within this customer's peer group, "
            "increasing relevance versus showing globally popular items to everyone."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.header("System Architecture")
    st.write(
        "End-to-end pipeline from raw Excel data through feature engineering, "
        "four parallel ML analyses, and a Streamlit dashboard with business insights."
    )

    arch_path = os.path.join(ROOT, "data", "architecture_diagram.png")
    if os.path.exists(arch_path):
        st.image(arch_path, use_container_width=True)
    else:
        with st.spinner("Generating diagram..."):
            arch_fig = generate_diagram(save_path=arch_path)
        st.pyplot(arch_fig)
        plt.close(arch_fig)

    st.markdown("""
    | Layer | Files | Key output |
    |---|---|---|
    | Data ingestion | `data_loader.py` | Raw DataFrame |
    | Cleaning | `data_cleaning.py` | 797K clean rows |
    | Feature engineering | `analytics.py` | RFM scores (1–5 quintiles) |
    | Segmentation | `segmentation.py` | K-Means clusters, PCA plot, silhouette 0.528 |
    | Churn modelling | `churn_model.py` | XGBoost, AUC 0.778, SHAP values |
    | Segment dynamics | `markov_analysis.py` | Transition matrix, steady-state distribution |
    | Retention | `cohort_analysis.py` | Month-1 retention 22.5% |
    | Recommendations | `recommendation.py` | Cluster-aware suggestions, Precision@10 |
    | Dashboard | `dashboard/app.py` | Streamlit, 7 tabs, live predictor |
    """)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("About")
    st.write("**Dataset:** UCI Online Retail II")
    st.write("**Customers:** 5,878")
    st.write("**Transactions:** ~1M rows")
    st.write("**Period:** Dec 2009 – Dec 2011")
    st.divider()
    st.title("RFM Segments")
    st.write("🏆 **Champions** — Recent, frequent, high spend")
    st.write("✅ **Loyal** — Regular buyers, solid spend")
    st.write("⚠️ **At Risk** — Haven't bought recently")
    st.write("💀 **Lost** — Long inactive, low engagement")
    st.divider()
    st.title("Model")
    st.write("**Algorithm:** XGBoost Classifier")
    st.write("**ROC-AUC:** 0.778")
    st.write("**Churn definition:** 180+ days inactive")
    st.write("**Explainability:** SHAP TreeExplainer")
