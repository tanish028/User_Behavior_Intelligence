import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

st.set_page_config(page_title="User Behavior Intelligence",
                   page_icon="PS", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@keyframes fadeInUp {
  from { opacity:0; transform:translateY(16px); }
  to   { opacity:1; transform:translateY(0); }
}
.main .block-container { animation:fadeInUp 0.45s ease-out; padding-top:1.5rem; }
[data-testid="metric-container"] {
  background:linear-gradient(135deg,#F8FAFC 0%,#EFF6FF 100%);
  border:1px solid #DBEAFE; border-radius:0.75rem; padding:1rem 1.2rem;
  transition:transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="metric-container"]:hover {
  transform:translateY(-3px); box-shadow:0 6px 18px rgba(37,99,235,0.12);
}
.stTabs [data-baseweb="tab-list"] {
  gap:4px; background:#F1F5F9; border-radius:0.6rem; padding:4px;
}
.stTabs [data-baseweb="tab"] {
  border-radius:0.5rem; padding:0.4rem 1rem;
  transition:background 0.2s ease, color 0.2s ease;
}
.streamlit-expanderHeader { transition:background-color 0.2s ease; border-radius:0.5rem; }
.streamlit-expanderHeader:hover { background-color:#EFF6FF !important; }
.kf-grid { display:flex; gap:1rem; margin-bottom:1.2rem; flex-wrap:wrap; }
.kf-card {
  flex:1; min-width:160px; background:#fff; border-radius:0.75rem;
  padding:1rem 1.2rem; border-left:4px solid;
  box-shadow:0 2px 8px rgba(0,0,0,0.06); animation:fadeInUp 0.5s ease-out both;
}
.kf-card.blue  { border-color:#2563EB; }
.kf-card.green { border-color:#10B981; }
.kf-card.amber { border-color:#F59E0B; }
.kf-card.red   { border-color:#EF4444; }
.kf-card h2 { margin:0 0 4px; font-size:1.7rem; font-weight:700; }
.kf-card p  { margin:0; font-size:0.82rem; color:#64748B; line-height:1.4; }
.sql-box {
  background:#1E293B; color:#E2E8F0; border-radius:0.6rem;
  padding:1rem 1.2rem; font-family:monospace; font-size:0.82rem;
  white-space:pre; overflow-x:auto; margin-bottom:0.8rem;
  border-left:3px solid #2563EB;
}
</style>
""", unsafe_allow_html=True)

# ── Style helpers ──────────────────────────────────────────────────────────────
BG, BORDER = "#F8FAFC", "#334155"
PALETTE = ["#2563EB","#10B981","#F59E0B","#EF4444","#8B5CF6",
           "#06B6D4","#EC4899","#84CC16","#F97316","#6366F1"]

def _style_ax(ax, title="", xlabel="", ylabel="", grid_axis="y"):
    ax.set_facecolor(BG)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#CBD5E1")
    ax.tick_params(colors="#475569", labelsize=9)
    if title:  ax.set_title(title, fontsize=12, fontweight="bold", color=BORDER, pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9, color="#475569")
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color="#475569")
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E2E8F0", lw=0.7, linestyle="--")
        ax.set_axisbelow(True)

def _fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    return fig, ax

# ── Imports ────────────────────────────────────────────────────────────────────
from recommendation import recommend_for_customer, recommend_popular, evaluate_recommendations
from cohort_analysis import build_cohort_matrix, plot_cohort_heatmap
from churn_model import (create_churn_label, train_churn_model, evaluate_model,
                         plot_feature_importance, predict_churn,
                         compute_shap_values, plot_shap_summary, plot_shap_waterfall)
from markov_analysis import (compute_monthly_segments, build_transition_matrix,
                              plot_transition_heatmap, compute_steady_state)
from architecture_diagram import generate_diagram
from sql_analytics import create_db, run_query, QUERIES
from forecasting import (prepare_monthly_revenue, decompose_and_forecast,
                         plot_forecast, plot_seasonal_pattern)
from clv_model import compute_clv, plot_clv_by_segment, plot_clv_vs_churn

# ── Cache ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df  = pd.read_csv(os.path.join(ROOT,"data","cleaned_data.csv"), parse_dates=["InvoiceDate"])
    rfm = pd.read_csv(os.path.join(ROOT,"data","rfm_data.csv"), index_col="CustomerID")
    return df, rfm

@st.cache_data
def get_cohort_data(_df):     return build_cohort_matrix(_df)

@st.cache_data
def get_markov_data(_df):
    ms = compute_monthly_segments(_df)
    tm = build_transition_matrix(ms)
    return tm, compute_steady_state(tm)

@st.cache_resource
def train_model(_rfm, _version=3):
    rfm_l = create_churn_label(_rfm, recency_threshold=180)
    model, X_test, y_test, features = train_churn_model(rfm_l)
    return model, X_test, y_test, features, rfm_l

@st.cache_data
def get_rec_metrics(_df, _rfm):
    return evaluate_recommendations(_df, _rfm, k=10, sample_size=200)

@st.cache_data
def get_forecast(_df):
    monthly = prepare_monthly_revenue(_df)
    return monthly, *decompose_and_forecast(monthly, periods_ahead=6)

@st.cache_data
def get_clv_data(_df, _rfm):
    model, _, _, _, _ = train_model(_rfm)
    return compute_clv(_df, _rfm, model)

df, rfm = load_data()
df["Month"]     = df["InvoiceDate"].dt.to_period("M").astype(str)
df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
df["Hour"]      = df["InvoiceDate"].dt.hour

# ── Header + KPIs ──────────────────────────────────────────────────────────────
st.title("PS User Behavior Intelligence")
st.caption("End-to-end customer analytics - UCI Online Retail II - 1M+ transactions - 5,878 customers - 2009-2011")
k1,k2,k3,k4 = st.columns(4)
k1.metric("Customers",    f"{rfm.shape[0]:,}")
k2.metric("Total Revenue",f"PS{df['TotalPrice'].sum():,.0f}")
k3.metric("Total Orders", f"{df['Invoice'].nunique():,}")
k4.metric("Countries",    f"{df['Country'].nunique()}")
st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tabs = st.tabs(["PS Overview","PS SQL Analytics","PS Geographic",
                "PS Cohort Retention","PS Customer Journeys",
                "PS Forecasting","PS CLV","PS Churn Prediction",
                "PS Recommendations","PS Architecture"])
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10 = tabs

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 - OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    churn_pct = round((rfm["Segment"].isin(["At Risk","Lost"])).mean()*100, 1)
    monthly_s = df.groupby("Month")["TotalPrice"].sum()
    peak_m    = monthly_s.idxmax()

    st.markdown(f"""
    <div class="kf-grid">
      <div class="kf-card blue"><h2>{rfm.shape[0]:,}</h2>
        <p>Unique customers across<br>41 countries, 2009-2011</p></div>
      <div class="kf-card red"><h2>{churn_pct}%</h2>
        <p>Customers in At-Risk or<br>Lost segments right now</p></div>
      <div class="kf-card green"><h2>PS{df['TotalPrice'].sum()/1e6:.1f}M</h2>
        <p>Total revenue generated<br>across the full period</p></div>
      <div class="kf-card amber"><h2>{peak_m}</h2>
        <p>Peak revenue month -<br>Q4 seasonal spike every year</p></div>
    </div>""", unsafe_allow_html=True)

    st.subheader("Customer Segments")
    c1,c2 = st.columns(2)
    with c1:
        seg = rfm["Segment"].value_counts()
        fig,ax = _fig(5,3.5)
        bars = ax.barh(seg.index, seg.values,
                       color=["#2563EB","#10B981","#F59E0B","#EF4444"][:len(seg)],
                       edgecolor="white", height=0.6)
        for b,v in zip(bars,seg.values):
            ax.text(b.get_width()+8,b.get_y()+b.get_height()/2,f"{v:,}",va="center",fontsize=9,color=BORDER)
        _style_ax(ax,"RFM Segments","Customers","",grid_axis="x")
        ax.set_xlim(0,seg.max()*1.18); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    with c2:
        cl = rfm["Cluster_Name"].value_counts()
        fig,ax = _fig(5,3.5)
        bars = ax.barh(cl.index, cl.values,
                       color=["#8B5CF6","#06B6D4","#EC4899"][:len(cl)],
                       edgecolor="white", height=0.6)
        for b,v in zip(bars,cl.values):
            ax.text(b.get_width()+8,b.get_y()+b.get_height()/2,f"{v:,}",va="center",fontsize=9,color=BORDER)
        _style_ax(ax,"K-Means Clusters","Customers","",grid_axis="x")
        ax.set_xlim(0,cl.max()*1.18); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.info("**PS Insight:** Most customers fall into At Risk or Lost - normal for e-commerce with high one-time buyer rates. The High Value cluster represents wholesale/B2B buyers who buy rarely but in very large quantities.")
    st.divider()
    st.subheader("Sales Trends")
    fig,ax = _fig(11,3.8)
    xv = range(len(monthly_s))
    ax.fill_between(xv,monthly_s.values,alpha=0.12,color="#2563EB")
    ax.plot(xv,monthly_s.values,color="#2563EB",lw=2,marker="o",markersize=3.5,
            markerfacecolor="white",markeredgewidth=1.2)
    step = max(1,len(monthly_s)//10)
    ax.set_xticks(list(xv)[::step]); ax.set_xticklabels(list(monthly_s.index)[::step],rotation=35,ha="right",fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:f"PS{v/1000:.0f}K"))
    _style_ax(ax,"Monthly Revenue","","Revenue"); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    c1,c2 = st.columns(2)
    with c1:
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        day_rev   = df.groupby("DayOfWeek")["TotalPrice"].sum().reindex(day_order)
        fig,ax    = _fig(5,3.5)
        colors    = ["#2563EB" if d!="Sunday" else "#EF4444" for d in day_order]
        ax.bar(day_order,day_rev.values,color=colors,edgecolor="white",width=0.6)
        ax.set_xticklabels([d[:3] for d in day_order],fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:f"PS{v/1000:.0f}K"))
        _style_ax(ax,"Revenue by Day of Week","","Revenue"); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    with c2:
        hr  = df.groupby("Hour")["TotalPrice"].sum()
        fig,ax = _fig(5,3.5)
        peak_h = hr.nlargest(3).index
        hc = ["#F59E0B" if h in peak_h else "#2563EB" for h in hr.index]
        ax.bar(hr.index,hr.values,color=hc,edgecolor="white",width=0.75)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:f"PS{v/1000:.0f}K"))
        _style_ax(ax,"Revenue by Hour (amber=peak)","Hour","Revenue"); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.info("**PS Insight:** Revenue spikes Oct-Nov each year (seasonal gift buying). Orders peak 10AM-3PM weekdays, near-zero on Sundays - consistent with a B2B wholesale customer base.")
    st.warning("**PS Data quality:** ~24% of raw transactions had no Customer ID and were excluded from RFM analysis.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 - SQL ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("SQL Analytics Layer")
    st.write("Transaction and RFM data loaded into an in-memory SQLite database. "
             "Each query demonstrates a different SQL concept — JOINs, window functions, "
             "subqueries, GROUP BY, ORDER BY. The same queries would run unchanged on "
             "PostgreSQL or BigQuery.")

    @st.cache_resource
    def get_db(_df, _rfm):
        return create_db(_df, _rfm)

    conn = get_db(df, rfm)

    query_name = st.selectbox("Select a query to run", list(QUERIES.keys()))
    q          = QUERIES[query_name]

    st.markdown(f'<div class="sql-box">{q["sql"].strip()}</div>', unsafe_allow_html=True)

    result = run_query(conn, q["sql"])
    st.dataframe(result, use_container_width=True)
    st.info(f"**PS Insight:** {q['insight']}")
    st.caption(f"{len(result):,} rows returned")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 - GEOGRAPHIC
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Geographic Revenue Breakdown")
    cr  = (df[df["Country"]!="Unspecified"].groupby("Country")["TotalPrice"].sum().sort_values(ascending=False))
    uk  = cr.get("United Kingdom",0); non_uk = cr.sum()-uk
    t10 = cr.head(10)
    c1,c2 = st.columns([2,1])
    with c1:
        fig,ax = _fig(8,4.5)
        bars = ax.barh(t10.index[::-1],t10.values[::-1],
                       color=["#EF4444" if c=="United Kingdom" else "#2563EB" for c in t10.index[::-1]],
                       edgecolor="white",height=0.65)
        for b,v in zip(bars,t10.values[::-1]):
            ax.text(b.get_width()+t10.max()*0.01,b.get_y()+b.get_height()/2,
                    f"PS{v/1000:.0f}K",va="center",fontsize=8.5,color=BORDER)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:f"PS{v/1e6:.1f}M"))
        _style_ax(ax,"Top 10 Countries by Revenue (red=UK)","","",grid_axis="x")
        ax.set_xlim(0,t10.max()*1.18); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    with c2:
        st.metric("UK Revenue",f"PS{uk:,.0f}",f"{uk/cr.sum()*100:.1f}% of total")
        st.metric("International Revenue",f"PS{non_uk:,.0f}",f"{non_uk/cr.sum()*100:.1f}% of total")
        st.metric("Countries Served",str(df["Country"].nunique()))
    st.info("**PS Insight:** UK is ~85% of revenue - significant concentration risk. Netherlands, Ireland, Germany, France are the top international markets for targeted expansion.")
    st.warning("**PS Concentration risk:** 85% tied to one country means any UK disruption has outsized business impact.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 - COHORT RETENTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Cohort Retention Analysis")
    st.write("Groups customers by first purchase month and tracks what % returned each subsequent month.")
    cohort_pct,_ = get_cohort_data(df)
    max_p = st.slider("Months to display",3,12,12)
    fig   = plot_cohort_heatmap(cohort_pct, max_periods=max_p)
    st.pyplot(fig); plt.close(fig)
    avg_m1 = cohort_pct[1].mean() if 1 in cohort_pct.columns else 0
    avg_m3 = cohort_pct[3].mean() if 3 in cohort_pct.columns else 0
    avg_m6 = cohort_pct[6].mean() if 6 in cohort_pct.columns else 0
    c1,c2,c3 = st.columns(3)
    c1.metric("Avg Month-1 Retention",f"{avg_m1:.1f}%")
    c2.metric("Avg Month-3 Retention",f"{avg_m3:.1f}%")
    c3.metric("Avg Month-6 Retention",f"{avg_m6:.1f}%")
    st.error(f"**PS Key finding:** Only {avg_m1:.1f}% of new customers return in month 1 - 3 in 4 are lost immediately. A 10-point improvement here outweighs acquiring 10% more new customers.")
    st.info("**PS Action:** Trigger an automated follow-up within 7 days of first purchase. Customers who return once become significantly more loyal.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 - CUSTOMER JOURNEYS (MARKOV)
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Customer Segment Transitions - Markov Chain")
    st.write("Tracks how customers move between RFM segments month-to-month and projects long-run distribution.")
    with st.spinner("Computing transition matrix..."):
        tm, steady = get_markov_data(df)
    c1,c2 = st.columns([2,1])
    with c1:
        mfig = plot_transition_heatmap(tm); st.pyplot(mfig); plt.close(mfig)
        st.caption("Each cell = probability of moving from row segment to column segment next month.")
    with c2:
        st.subheader("Steady State")
        for seg,pct in steady.items(): st.metric(seg,f"{pct:.1f}%")
    st.error(f"**PS Key finding:** {steady.get('Lost',0):.1f}% of customers will eventually reach Lost if no intervention occurs.")
    st.info(f"**PS Insight:** Champions stabilise at {steady.get('Champions',0):.1f}% long-run. Intercept the Loyal - At Risk transition before customers reach Lost.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 - FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("Revenue Forecasting")
    st.write("Classical time-series decomposition: linear trend + seasonal indices, implemented from scratch in numpy. "
             "Forecasts 6 months ahead with 95% confidence bands.")

    with st.spinner("Running forecast..."):
        monthly, trend_vals, seasonal_idx, fitted, forecast, lower, upper, res_std = get_forecast(df)

    slope = float(np.polyfit(range(len(monthly)), monthly.values, 1)[0])
    c1,c2,c3 = st.columns(3)
    c1.metric("Monthly Trend",f"PS{slope:+,.0f}/mo",help="Linear trend slope")
    c2.metric("Next Month Forecast",f"PS{forecast.iloc[0]:,.0f}")
    c3.metric("Forecast Uncertainty",f"PS{res_std:,.0f}",help="1-sigma residual std")

    fig = plot_forecast(monthly, fitted, forecast, lower, upper)
    st.pyplot(fig); plt.close(fig)
    st.info("**PS How it works:** Step 1 - fit a linear trend using OLS. Step 2 - compute average monthly deviation from trend (seasonal index). Step 3 - project trend forward and add seasonal adjustment. Confidence band = 1.96 x historical residual std.")

    with st.expander("PS Seasonal Pattern - which months run above/below trend"):
        fig2 = plot_seasonal_pattern(seasonal_idx)
        st.pyplot(fig2); plt.close(fig2)
        st.info("**PS Insight:** Green months run above trend (higher than expected), red below. This shows the seasonal calendar for planning inventory and marketing spend.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 - CLV
# ═══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.header("Customer Lifetime Value (CLV)")
    st.write("Predicted CLV connects directly to the churn model: "
             "CLV = avg monthly spend x expected months remaining, "
             "where expected months = 1 / monthly churn probability (geometric distribution).")

    with st.spinner("Computing CLV for all customers..."):
        clv_df = get_clv_data(df, rfm)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Avg CLV (all)",      f"PS{clv_df['predicted_clv'].mean():,.0f}")
    c2.metric("Median CLV",         f"PS{clv_df['predicted_clv'].median():,.0f}")
    c3.metric("Top 10% CLV cutoff", f"PS{clv_df['predicted_clv'].quantile(0.9):,.0f}")
    c4.metric("Total Projected CLV",f"PS{clv_df['predicted_clv'].sum():,.0f}")

    fig = plot_clv_by_segment(clv_df)
    st.pyplot(fig); plt.close(fig)
    st.info("**PS Insight:** Champions have the highest average CLV by a large margin. Even a small reduction in Champions churn rate has an outsized revenue impact compared to improving retention of Lost customers.")

    st.divider()
    st.subheader("CLV vs Churn Risk - Who to prioritise?")
    st.write("The top-right quadrant (high CLV + high churn risk) is where retention spend should be focused. These customers will generate the most revenue AND are most likely to leave without intervention.")
    fig2 = plot_clv_vs_churn(clv_df)
    st.pyplot(fig2); plt.close(fig2)

    with st.expander("PS Top 20 Customers by Predicted CLV"):
        top20 = (clv_df[["total_spend","avg_monthly_spend","expected_months",
                          "churn_prob_annual","predicted_clv","Segment","Cluster_Name"]]
                 .sort_values("predicted_clv", ascending=False)
                 .head(20)
                 .rename(columns={"churn_prob_annual":"churn_prob",
                                  "expected_months":"exp_months",
                                  "predicted_clv":"clv"}))
        top20["churn_prob"] = top20["churn_prob"].map("{:.1%}".format)
        top20["clv"]        = top20["clv"].map("PS{:,.0f}".format)
        st.dataframe(top20, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 - CHURN PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab8:
    st.header("Churn Prediction - XGBoost")
    st.write("Churned = no purchase in 180+ days. Features: F-Score and M-Score only (R-Score excluded - data leakage).")

    model,X_test,y_test,features,rfm_l = train_model(rfm)
    auc,eval_fig = evaluate_model(model,X_test,y_test)
    c1,c2,c3 = st.columns(3)
    c1.metric("ROC-AUC",f"{auc:.3f}",help="0.5=random | 0.75+=solid")
    c2.metric("Churned Customers",f"{rfm_l['Churned'].sum():,}")
    c3.metric("Churn Rate",f"{rfm_l['Churned'].mean()*100:.1f}%")
    st.info(f"**PS What AUC {auc:.3f} means:** Pick one random churner and one active customer - the model ranks the churner as riskier {auc*100:.1f}% of the time. Random guessing = 50%.")

    with st.expander("PS Model Evaluation - Confusion Matrix and ROC Curve"):
        st.pyplot(eval_fig); plt.close(eval_fig)
    with st.expander("PS Feature Importance"):
        imp_fig = plot_feature_importance(model,features)
        st.pyplot(imp_fig); plt.close(imp_fig)
        st.info("**PS Insight:** F-Score drives ~91% of predictive power. How often a customer buys matters far more than how much they spend.")
    with st.expander("PS SHAP Explainability - Why was this customer flagged?"):
        with st.spinner("Computing SHAP values..."):
            explainer,shap_values = compute_shap_values(model,X_test)
        fig_s = plot_shap_summary(shap_values,X_test); st.pyplot(fig_s); plt.close(fig_s)
        cidx  = st.slider("Customer index (test set)",0,min(len(X_test)-1,50),0)
        fig_w = plot_shap_waterfall(explainer,shap_values,X_test,customer_idx=cidx)
        st.pyplot(fig_w); plt.close(fig_w)
        st.caption("PS Red = increased churn risk | PS Blue = reduced churn risk")

    st.divider()
    st.subheader("PS Live Churn Predictor")
    c1,c2 = st.columns(2)
    with c1: f = st.slider("F-Score",1,5,3,help="5=very frequent | 1=rarely buys")
    with c2: m = st.slider("M-Score",1,5,3,help="5=high spend | 1=low spend")
    prob = predict_churn(model,f,m); pct = prob*100
    if   pct>=70: st.error(f"**{pct:.1f}% churn risk** - High. Prioritise for re-engagement.")
    elif pct>=40: st.warning(f"**{pct:.1f}% churn risk** - Medium. Consider a targeted offer.")
    else:         st.success(f"**{pct:.1f}% churn risk** - Low. Healthy engagement signals.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9 - RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab9:
    st.header("Cluster-Aware Product Recommendations")
    st.write("Recommendations are tailored to the customer's K-Means cluster. Quality measured with Precision@10 and Recall@10 via leave-one-out evaluation.")
    with st.spinner("Evaluating recommendation quality..."):
        metrics = get_rec_metrics(df,rfm)
    k = 10
    c1,c2,c3 = st.columns(3)
    c1.metric(f"Precision@{k}",f"{metrics.get(f'Precision@{k}',0):.3f}")
    c2.metric(f"Recall@{k}",   f"{metrics.get(f'Recall@{k}',0):.3f}")
    c3.metric("Customers Evaluated",f"{metrics.get('n_evaluated',0):,}")
    p = metrics.get(f"Precision@{k}",0)
    st.info(f"**PS Precision@10 = {p:.3f}:** On average {p*10:.1f} of the 10 recommended items were things the customer actually bought in their next visit (leave-one-out evaluation).")
    st.divider()
    selected = st.selectbox("Select a Customer ID",rfm.index.tolist())
    if selected:
        cl = rfm.loc[selected,"Cluster_Name"]; sg = rfm.loc[selected,"Segment"]
        fs = int(rfm.loc[selected,"F-Score"]);  ms = int(rfm.loc[selected,"M-Score"])
        cp = predict_churn(model,fs,ms)
        c1,c2,c3 = st.columns(3)
        c1.metric("Segment",sg); c2.metric("Cluster",cl); c3.metric("Churn Risk",f"{cp*100:.1f}%")
        c1,c2 = st.columns(2)
        with c1:
            st.subheader("Global Popular Products")
            st.dataframe(recommend_popular(df).reset_index(),use_container_width=True)
        with c2:
            st.subheader(f"Recommended for {selected}")
            st.dataframe(recommend_for_customer(selected,df,rfm).reset_index(),use_container_width=True)
        st.info(f"**PS Insight:** Customer is in the {cl} cluster ({sg} segment). Cluster-aware recommendations filter to products popular within their peer group - more relevant than global popularity.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 10 - ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
with tab10:
    st.header("System Architecture")
    arch_path = os.path.join(ROOT,"data","architecture_diagram.png")
    if os.path.exists(arch_path):
        st.image(arch_path, use_container_width=True)
    else:
        with st.spinner("Generating diagram..."):
            af = generate_diagram(save_path=arch_path)
        st.pyplot(af); plt.close(af)
    st.markdown("""
| Layer | Files | Key output |
|---|---|---|
| Data ingestion | data_loader.py | Raw DataFrame |
| Cleaning | data_cleaning.py | 797K clean rows |
| Feature engineering | analytics.py | RFM scores 1-5 |
| SQL analytics | sql_analytics.py | 7 analytical queries on SQLite |
| Segmentation | segmentation.py | K-Means clusters, PCA, silhouette 0.528 |
| Forecasting | forecasting.py | 6-month revenue forecast (trend + seasonal) |
| CLV | clv_model.py | Per-customer lifetime value from churn model |
| Churn modelling | churn_model.py | XGBoost AUC 0.778, SHAP explainability |
| Segment dynamics | markov_analysis.py | Transition matrix, steady-state |
| Retention | cohort_analysis.py | Month-1 retention 22.5% |
| Recommendations | recommendation.py | Cluster-aware, Precision@10 evaluated |
| Dashboard | dashboard/app.py | Streamlit, 10 tabs, live predictors |
""")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("About")
    st.write("**Dataset:** UCI Online Retail II")
    st.write("**Customers:** 5,878 | **Transactions:** 1M+")
    st.write("**Period:** Dec 2009 - Dec 2011")
    st.divider()
    st.title("Segments")
    st.write("**Champions** - Recent, frequent, high spend")
    st.write("**Loyal** - Regular buyers, solid spend")
    st.write("**At Risk** - Not bought recently")
    st.write("**Lost** - Long inactive")
    st.divider()
    st.title("Models")
    st.write("**Churn:** XGBoost, AUC 0.778")
    st.write("**Forecast:** Trend + Seasonal decomposition")
    st.write("**CLV:** Geometric lifetime x monthly spend")
