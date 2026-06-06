# User Behavior Intelligence System

An end-to-end customer analytics pipeline built on the **UCI Online Retail II** dataset (~1M transactions, 5,878 customers, 2009–2011). The system cleans raw transactional data, engineers RFM features, segments customers via K-Means clustering, models segment transitions with Markov chains, predicts churn with XGBoost, and surfaces cluster-aware product recommendations — all through an interactive Streamlit dashboard.

**Live demo:** https://userbehaviorintelligence-fu4vi8vjfsvnxurefjeyen.streamlit.app/

---

## Problem Statement

Retailers accumulate large volumes of transactional data but rarely extract actionable customer intelligence from it. This project answers three business questions:

1. **Who are our customers?** — RFM-based segmentation (Champions, Loyal, At Risk, Lost)
2. **How do customers move between segments?** — Markov chain transition matrix
3. **Which customers might churn?** — XGBoost classifier (ROC-AUC: 0.778)
4. **How long do customers stay?** — Cohort retention analysis
5. **What should we recommend?** — Cluster-aware product recommendations

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | pandas, numpy |
| Machine learning | scikit-learn (KMeans, PCA, StandardScaler, silhouette score), XGBoost |
| Visualisation | matplotlib, seaborn |
| Dashboard | Streamlit |
| Data source | UCI Online Retail II (.xlsx via openpyxl) |

---

## Project Structure

```
user_behavior_intelligence/
├── data/
│   ├── online_retail_II.xlsx     # Raw dataset
│   ├── cleaned_data.csv          # Output of data_cleaning.py
│   ├── rfm_data.csv              # Output of segmentation.py
│   └── *.png                     # Generated plots
├── src/
│   ├── data_loader.py            # Loads raw Excel file
│   ├── data_cleaning.py          # Cleans and validates transactions
│   ├── analytics.py              # RFM computation + purchase pattern plots
│   ├── segmentation.py           # K-Means clustering + PCA + churn risk
│   ├── cohort_analysis.py        # Cohort retention matrix + heatmap
│   ├── markov_analysis.py        # Markov chain segment transition matrix
│   ├── churn_model.py            # XGBoost churn classifier + evaluation
│   └── recommendation.py        # Cluster-aware product recommendations
├── dashboard/
│   └── app.py                    # Streamlit dashboard
├── notebooks/
│   └── exploration.ipynb         # EDA notebook
└── requirements.txt
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the data pipeline (from project root)
python src/data_cleaning.py       # Generates cleaned_data.csv
python src/analytics.py           # Generates plots
python src/segmentation.py        # Generates rfm_data.csv with clusters

# 3. Launch dashboard
streamlit run dashboard/app.py
```

---

## Key Findings

- **3 customer segments** identified via K-Means (silhouette score reported at runtime):
  - **High Value** — infrequent but very high spend; likely bulk/wholesale buyers
  - **Active Regular** — moderate recency, consistent frequency
  - **Churned** — long inactive, low spend; high churn risk score
- **RFM scoring** places the majority of customers in "At Risk" or "Lost" — typical for retail datasets with high one-time buyer rates
- **Revenue** peaks in Q4 (Nov–Dec), consistent with seasonal retail patterns
- **Peak order hours**: 10 AM–3 PM on weekdays; Sunday is the quietest day

---

## ML Methodology

### Feature Engineering — RFM

| Feature | Definition |
|---|---|
| Recency | Days since last purchase (lower = better) |
| Frequency | Unique invoices per customer |
| Monetary | Total spend per customer |

Outliers capped at 99th percentile before scaling to reduce the influence of wholesale buyers on cluster centroids.

### Clustering

- **Algorithm**: K-Means with `StandardScaler` normalisation
- **K selection**: Elbow method (inertia) + silhouette score across k=2..10
- **Dimensionality reduction**: PCA (2 components) for cluster visualisation

### Churn Prediction (XGBoost)

**Label:** A customer is marked as churned if they have not purchased in 180+ days (`recency > 180`).

**Features:** F-Score and M-Score only. R-Score was initially included but produced an AUC of 0.996 — identified as **data leakage** (R-Score is a quintile of recency, which directly encodes the churn label). Removing R-Score gives an honest AUC of **0.778**.

**Model:** `XGBClassifier` with 100 trees, max depth 4, learning rate 0.1, and `scale_pos_weight` to handle class imbalance (40.8% churned).

**Evaluation:**

| Metric | Value |
|---|---|
| ROC-AUC | 0.778 |
| Dominant feature | F-Score (importance: 0.91) |
| Second feature | M-Score (importance: 0.09) |

**ROC-AUC interpretation:** Measures the probability that the model scores a randomly chosen churner higher than a randomly chosen non-churner. 0.5 = random guessing, 1.0 = perfect. 0.778 means the model correctly ranks churners above non-churners 77.8% of the time.

**Key insight:** Purchase frequency (F-Score) is a far stronger churn signal than monetary spend (M-Score). Customers who buy regularly are much harder to lose than those who made a single large purchase.

### Markov Chain Transition Analysis

- Computes each customer's RFM segment every month
- Builds a transition probability matrix: probability of moving from segment X to segment Y next month
- Computes the steady-state distribution using the power method (long-run segment proportions)

### Cohort Retention Analysis

- Groups customers by first purchase month
- Tracks what % of each cohort returned in months 1, 2, 3...
- Visualised as a colour-coded heatmap

### Churn Risk Score

Min-max normalised recency in [0, 1]. Score of 1.0 = highest churn risk.

---

## Deploying to Streamlit Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your repo → set **Main file path** to `dashboard/app.py`
4. Click **Deploy** — your dashboard will be live at a public URL in ~2 minutes

> **Note:** The pre-built `data/cleaned_data.csv` and `data/rfm_data.csv` files are committed to the repo, so no pipeline re-run is needed on deployment.

---

## Dataset

**Online Retail II** — UCI Machine Learning Repository  
https://archive.ics.uci.edu/dataset/502/online+retail+ii  
UK-based online retailer, Dec 2009 – Dec 2011, ~1M rows.
