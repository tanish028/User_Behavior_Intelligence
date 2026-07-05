"""
api/main.py  —  FastAPI service for User Behavior Intelligence
--------------------------------------------------------------
Exposes three ML capabilities as HTTP endpoints:
  1. POST /predict/churn        — churn probability for a customer
  2. GET  /customer/{id}        — full RFM profile + CLV + segment
  3. GET  /recommend/{id}       — cluster-aware product recommendations

Why FastAPI over Flask?
  - Automatic request validation via Pydantic (no manual if/else checks)
  - Auto-generated interactive docs at /docs (great for demos)
  - Type hints drive everything: parsing, validation, documentation

Run locally:
  uvicorn api.main:app --reload --port 8001
Then open: http://127.0.0.1:8001/docs
"""

import os
import sys
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── make src/ importable when running from project root ──────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))

from churn_model import create_churn_label, train_churn_model, predict_churn
from recommendation import recommend_for_customer, recommend_popular


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Shared state
# ─────────────────────────────────────────────────────────────────────────────
# We store everything in a plain dict called `ml`.
# Loading the model inside a request handler would be slow (100ms+ per call).
# Instead we load once at startup and keep it in memory for the life of the
# server process.
ml: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Lifespan (startup / shutdown)
# ─────────────────────────────────────────────────────────────────────────────
# FastAPI runs the code BEFORE `yield` when the server starts.
# Code AFTER `yield` runs when the server shuts down (cleanup).
# This replaces the old @app.on_event("startup") pattern.

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──────────────────────────────────────────────────────────────
    print("Loading data and model...")

    # Load RFM data (one row per customer)
    rfm = pd.read_csv(os.path.join(ROOT, "data", "rfm_data.csv"),
                      index_col="CustomerID")

    # Load transactions (needed for recommendations)
    df = pd.read_csv(os.path.join(ROOT, "data", "cleaned_data.csv"),
                     parse_dates=["InvoiceDate"])

    # Train churn model — create_churn_label adds the "Churned" column
    rfm_labelled = create_churn_label(rfm, recency_threshold=180)
    model, _, _, _ = train_churn_model(rfm_labelled)

    # Store everything in the shared dict
    ml["model"] = model
    ml["rfm"]   = rfm
    ml["df"]    = df

    print(f"Ready — {len(rfm):,} customers loaded, model trained.")

    yield   # <— server is now running and handling requests

    # ── SHUTDOWN ─────────────────────────────────────────────────────────────
    ml.clear()
    print("Shutdown: state cleared.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — App instance
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="User Behavior Intelligence API",
    description=(
        "ML-powered customer analytics API. "
        "Churn prediction via XGBoost, cluster-aware recommendations, "
        "and full RFM customer profiles."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware — lets a browser (or the Streamlit dashboard) call this API
# from a different port without being blocked by browser security rules.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production (list your domains)
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Pydantic schemas (request + response shapes)
# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models do two things:
#   a) validate incoming data before your function runs
#   b) document the expected shape in /docs automatically

class ChurnRequest(BaseModel):
    """Input schema for the churn prediction endpoint."""
    f_score: int = Field(..., ge=1, le=5,
                         description="Frequency score (1=rare buyer, 5=very frequent)")
    m_score: int = Field(..., ge=1, le=5,
                         description="Monetary score (1=low spend, 5=high spend)")


class ChurnResponse(BaseModel):
    """Output schema for churn prediction."""
    f_score: int
    m_score: int
    churn_probability: float = Field(..., description="Probability of churn (0–1)")
    risk_level: str          = Field(..., description="Low / Medium / High")
    action: str              = Field(..., description="Recommended business action")


class CustomerProfile(BaseModel):
    """Full customer profile returned by /customer/{id}."""
    customer_id: int
    recency_days: float
    frequency: int
    monetary: float
    r_score: int
    f_score: int
    m_score: int
    segment: str
    cluster_name: str
    churn_probability: float
    risk_level: str


class RecommendationResponse(BaseModel):
    """Product recommendation response."""
    customer_id: int
    cluster_name: str
    recommendations: list[str]   # list of product description strings


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Helper: map probability to risk level + action
# ─────────────────────────────────────────────────────────────────────────────
def _risk(prob: float) -> tuple[str, str]:
    """Return (risk_level, action) given a churn probability."""
    if prob >= 0.70:
        return "High",   "Trigger immediate re-engagement campaign with discount offer"
    if prob >= 0.40:
        return "Medium", "Send personalised email with new product recommendations"
    return     "Low",    "Customer is healthy — maintain regular comms"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Endpoints
# ─────────────────────────────────────────────────────────────────────────────

# ── 6a. Health check ─────────────────────────────────────────────────────────
# Every production API needs this. Load balancers ping it to check the service
# is alive. Also useful as a quick sanity check after deployment.

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms API is running and data is loaded."""
    return {
        "status":    "ok",
        "api":       "User Behavior Intelligence",
        "version":   "1.0.0",
        "customers": len(ml.get("rfm", [])),
        "docs":      "/docs",
    }


# ── 6b. Churn prediction ─────────────────────────────────────────────────────
# POST because we're sending a body (the customer scores), not a simple URL.
# response_model= tells FastAPI to validate the return value against ChurnResponse
# and strip any extra keys — good practice.

@app.post("/predict/churn", response_model=ChurnResponse, tags=["Prediction"])
def predict_churn_endpoint(data: ChurnRequest):
    """
    Predict churn probability for a customer given their F and M scores.

    R-Score is intentionally excluded — it is derived from recency which
    directly defines the churn label, so including it would be data leakage.
    """
    # predict_churn is our existing function from churn_model.py
    prob = predict_churn(ml["model"], data.f_score, data.m_score)
    risk, action = _risk(prob)

    # FastAPI validates this dict against ChurnResponse before sending it
    return {
        "f_score":           data.f_score,
        "m_score":           data.m_score,
        "churn_probability": round(float(prob), 4),
        "risk_level":        risk,
        "action":            action,
    }


# ── 6c. Customer profile ─────────────────────────────────────────────────────
# GET because we're just reading data — no side effects.
# {customer_id} is a path parameter: /customer/12345

@app.get("/customer/{customer_id}", response_model=CustomerProfile, tags=["Customer"])
def get_customer(customer_id: int):
    """
    Return the full RFM profile, segment, cluster, and churn risk for a customer.

    Raises 404 if the customer ID is not in the dataset.
    """
    rfm = ml["rfm"]

    # Check the customer exists — always validate before accessing data
    if customer_id not in rfm.index:
        raise HTTPException(
            status_code=404,
            detail=f"Customer {customer_id} not found in dataset"
        )

    row  = rfm.loc[customer_id]
    prob = predict_churn(ml["model"], int(row["F-Score"]), int(row["M-Score"]))
    risk, _ = _risk(prob)

    return {
        "customer_id":       customer_id,
        "recency_days":      round(float(row["recency"]),   1),
        "frequency":         int(row["frequency"]),
        "monetary":          round(float(row["monetary"]),  2),
        "r_score":           int(row["R-Score"]),
        "f_score":           int(row["F-Score"]),
        "m_score":           int(row["M-Score"]),
        "segment":           str(row["Segment"]),
        "cluster_name":      str(row["Cluster_Name"]),
        "churn_probability": round(float(prob), 4),
        "risk_level":        risk,
    }


# ── 6d. Recommendations ──────────────────────────────────────────────────────
# n is a query parameter: /recommend/12345?n=5
# It has a default value of 5 so it's optional.

@app.get("/recommend/{customer_id}", response_model=RecommendationResponse,
         tags=["Recommendations"])
def get_recommendations(customer_id: int, n: int = 5):
    """
    Return top-N cluster-aware product recommendations for a customer.

    Products are ranked by total revenue within the customer's K-Means cluster,
    so recommendations reflect what similar customers actually bought.

    - **n**: Number of recommendations to return (default 5, max 20)
    """
    # Guard against silly n values — Field validators would also work here
    if n < 1 or n > 20:
        raise HTTPException(status_code=422,
                            detail="n must be between 1 and 20")

    rfm = ml["rfm"]
    if customer_id not in rfm.index:
        raise HTTPException(status_code=404,
                            detail=f"Customer {customer_id} not found in dataset")

    cluster = str(rfm.loc[customer_id, "Cluster_Name"])

    # recommend_for_customer returns a pd.Series: index=product, values=revenue
    recs = recommend_for_customer(customer_id, ml["df"], rfm, n=n)
    products = recs.index.tolist()[:n]

    return {
        "customer_id":    customer_id,
        "cluster_name":   cluster,
        "recommendations": products,
    }
