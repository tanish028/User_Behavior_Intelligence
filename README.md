# User Behavior Intelligence System

An end-to-end customer analytics pipeline built on the **UCI Online Retail II** dataset (~1M transactions, 5,878 customers, 2009–2011). The system cleans raw transactional data, engineers RFM features, segments customers via K-Means clustering, scores churn risk, and surfaces cluster-aware product recommendations through an interactive Streamlit dashboard.

**Live demo:** https://userbehaviorintelligence-fu4vi8vjfsvnxurefjeyen.streamlit.app/

---

## Problem Statement

Retailers accumulate large volumes of transactional data but rarely extract actionable customer intelligence from it. This project answers three business questions:

1. **Who are our customers?** — RFM-based segmentation (Champions, Loyal, At Risk, Lost)
2. **Which customers might churn?** — Recency-based churn risk scoring
3. **What should we recommend?** — Cluster-aware product recommendations

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | pandas, numpy |
| Machine learning | scikit-learn (KMeans, PCA, StandardScaler, silhouette score) |
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
