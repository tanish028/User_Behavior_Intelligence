"""
sql_analytics.py
Loads transaction and RFM data into an in-memory SQLite database and
exposes a set of analytical queries that demonstrate SQL skills.

Why SQLite?
  - Zero setup: no server, no config, runs anywhere
  - In-memory: nothing written to disk, safe for demos
  - Real SQL: same syntax as PostgreSQL/MySQL for GROUP BY, JOINs, subqueries
"""
import sqlite3
import pandas as pd


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def create_db(df: pd.DataFrame, rfm: pd.DataFrame) -> sqlite3.Connection:
    """
    Load transactions and RFM into an in-memory SQLite database.

    Tables created:
      - transactions  : one row per line item (CustomerID, Invoice, Description, ...)
      - rfm           : one row per customer (RFM scores, segment, cluster)

    Returns:
        conn : open sqlite3 connection (caller must close when done)
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    df.to_sql("transactions", conn, index=False, if_exists="replace")
    rfm.reset_index().to_sql("rfm", conn, index=False, if_exists="replace")
    return conn


def run_query(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    """Execute a SQL query and return the result as a DataFrame."""
    return pd.read_sql_query(sql, conn)


# ---------------------------------------------------------------------------
# Analytical queries
# ---------------------------------------------------------------------------

QUERIES = {
    "Monthly Revenue Trend": {
        "sql": """
            SELECT
                strftime('%Y-%m', InvoiceDate)  AS month,
                ROUND(SUM(TotalPrice), 2)        AS revenue,
                COUNT(DISTINCT Invoice)          AS num_orders,
                COUNT(DISTINCT CustomerID)       AS active_customers
            FROM transactions
            WHERE CustomerID IS NOT NULL
            GROUP BY month
            ORDER BY month
        """,
        "insight": (
            "Revenue peaks every Oct-Nov (seasonal gift buying). "
            "Active customers and order count move together, confirming "
            "the seasonality is driven by existing customers returning, "
            "not just a spike in new acquisitions."
        ),
    },

    "Top 10 Customers by Total Spend": {
        "sql": """
            SELECT
                t.CustomerID,
                r.Segment,
                r."Cluster_Name"                       AS cluster,
                ROUND(SUM(t.TotalPrice), 2)            AS total_spend,
                COUNT(DISTINCT t.Invoice)              AS num_orders,
                ROUND(SUM(t.TotalPrice)
                      / COUNT(DISTINCT t.Invoice), 2)  AS avg_order_value
            FROM transactions t
            JOIN rfm r ON t.CustomerID = r.CustomerID
            WHERE t.CustomerID IS NOT NULL
            GROUP BY t.CustomerID, r.Segment, r."Cluster_Name"
            ORDER BY total_spend DESC
            LIMIT 10
        """,
        "insight": (
            "Top customers are overwhelmingly in the High Value cluster "
            "with very high average order values — consistent with the "
            "B2B/wholesale buyer hypothesis. These 10 customers likely "
            "account for a disproportionate share of total revenue."
        ),
    },

    "Revenue and Orders by RFM Segment": {
        "sql": """
            SELECT
                r.Segment,
                COUNT(DISTINCT t.CustomerID)           AS customers,
                ROUND(SUM(t.TotalPrice), 2)            AS total_revenue,
                COUNT(DISTINCT t.Invoice)              AS total_orders,
                ROUND(SUM(t.TotalPrice)
                      / COUNT(DISTINCT t.CustomerID), 2) AS revenue_per_customer
            FROM transactions t
            JOIN rfm r ON t.CustomerID = r.CustomerID
            WHERE t.CustomerID IS NOT NULL
            GROUP BY r.Segment
            ORDER BY total_revenue DESC
        """,
        "insight": (
            "Champions generate the most revenue per customer by a wide margin. "
            "Lost customers have low revenue per customer but may still hold "
            "re-engagement value if targeted with the right offer."
        ),
    },

    "Average Order Value by K-Means Cluster": {
        "sql": """
            SELECT
                r."Cluster_Name"                        AS cluster,
                COUNT(DISTINCT t.CustomerID)            AS customers,
                COUNT(DISTINCT t.Invoice)               AS orders,
                ROUND(SUM(t.TotalPrice), 2)             AS total_revenue,
                ROUND(AVG(t.TotalPrice), 2)             AS avg_line_value,
                ROUND(SUM(t.TotalPrice)
                      / COUNT(DISTINCT t.Invoice), 2)   AS avg_order_value
            FROM transactions t
            JOIN rfm r ON t.CustomerID = r.CustomerID
            WHERE t.CustomerID IS NOT NULL
            GROUP BY r."Cluster_Name"
            ORDER BY avg_order_value DESC
        """,
        "insight": (
            "The High Value cluster has a dramatically higher average order value "
            "than other clusters — this is the wholesale/B2B segment buying in bulk. "
            "Marketing strategy should differ significantly by cluster."
        ),
    },

    "Top 15 Products by Revenue": {
        "sql": """
            SELECT
                Description                             AS product,
                ROUND(SUM(TotalPrice), 2)               AS total_revenue,
                SUM(Quantity)                           AS units_sold,
                COUNT(DISTINCT Invoice)                 AS num_orders,
                ROUND(AVG(UnitPrice), 2)                AS avg_unit_price
            FROM transactions
            WHERE Description NOT IN (
                'POSTAGE','MANUAL','DOTCOM POSTAGE','CRUK COMMISSION'
            )
            AND CustomerID IS NOT NULL
            GROUP BY Description
            ORDER BY total_revenue DESC
            LIMIT 15
        """,
        "insight": (
            "A small number of products drive a large share of revenue — "
            "classic Pareto distribution. These should be priority items for "
            "stock management and promotional campaigns."
        ),
    },

    "Customer Retention Rate": {
        "sql": """
            SELECT
                total_customers,
                returning_customers,
                ROUND(100.0 * returning_customers / total_customers, 1)
                    AS retention_pct,
                one_time_buyers,
                ROUND(100.0 * one_time_buyers / total_customers, 1)
                    AS one_time_pct
            FROM (
                SELECT
                    COUNT(DISTINCT CustomerID)                              AS total_customers,
                    COUNT(DISTINCT CASE WHEN orders >= 2
                                   THEN CustomerID END)                    AS returning_customers,
                    COUNT(DISTINCT CASE WHEN orders = 1
                                   THEN CustomerID END)                    AS one_time_buyers
                FROM (
                    SELECT CustomerID, COUNT(DISTINCT Invoice) AS orders
                    FROM transactions
                    WHERE CustomerID IS NOT NULL
                    GROUP BY CustomerID
                )
            )
        """,
        "insight": (
            "One-time buyers represent the largest segment of the customer base. "
            "Improving retention from first to second purchase is the single "
            "highest-leverage growth opportunity — consistent with the cohort analysis."
        ),
    },

    "Revenue by Country (Top 10)": {
        "sql": """
            SELECT
                Country,
                COUNT(DISTINCT CustomerID)             AS customers,
                COUNT(DISTINCT Invoice)                AS orders,
                ROUND(SUM(TotalPrice), 2)              AS revenue,
                ROUND(100.0 * SUM(TotalPrice)
                      / SUM(SUM(TotalPrice)) OVER (), 1) AS revenue_share_pct
            FROM transactions
            WHERE Country != 'Unspecified'
            AND CustomerID IS NOT NULL
            GROUP BY Country
            ORDER BY revenue DESC
            LIMIT 10
        """,
        "insight": (
            "The UK dominates at ~85% revenue share — significant concentration risk. "
            "Note the use of a window function (SUM OVER ()) to compute share "
            "without a subquery, which is more efficient on large datasets."
        ),
    },
}


# ---------------------------------------------------------------------------
# Main — run all queries and print results
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df  = pd.read_csv("data/cleaned_data.csv", parse_dates=["InvoiceDate"])
    rfm = pd.read_csv("data/rfm_data.csv", index_col="CustomerID")

    conn = create_db(df, rfm)
    for title, q in QUERIES.items():
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        print(run_query(conn, q["sql"]).to_string(index=False))
    conn.close()
