"""
forecasting.py
Revenue forecasting using classical time-series decomposition:
  Trend    — ordinary least-squares linear fit (numpy.polyfit)
  Seasonal — average monthly deviation from trend (12-month indices)
  Forecast — trend projection + seasonal adjustment + confidence band

No external forecasting library required: pure numpy + pandas.
Implementing decomposition manually demonstrates understanding of the
underlying mechanics rather than just calling statsmodels/Prophet.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Core forecasting
# ---------------------------------------------------------------------------

def prepare_monthly_revenue(df: pd.DataFrame) -> pd.Series:
    """Aggregate transactions to a monthly revenue Series."""
    monthly = (
        df.assign(Month=df["InvoiceDate"].dt.to_period("M"))
        .groupby("Month")["TotalPrice"]
        .sum()
        .sort_index()
    )
    monthly.index = monthly.index.to_timestamp()
    return monthly


def decompose_and_forecast(monthly: pd.Series, periods_ahead: int = 6):
    """
    Decomposes monthly revenue into trend + seasonal components,
    then forecasts `periods_ahead` months into the future.

    Steps:
      1. Fit a linear trend using OLS (numpy.polyfit on integer time index)
      2. Compute seasonal indices: average (actual - trend) by calendar month
      3. Reconstruct fitted values = trend + seasonal
      4. Compute residual std for confidence bands
      5. Project trend forward + apply seasonal indices

    Returns:
        trend_vals   : np.array of trend values for historical period
        seasonal_idx : dict of {month_int: seasonal_offset}
        fitted       : pd.Series of fitted (trend + seasonal) values
        forecast     : pd.Series of forecasted values (next `periods_ahead` months)
        lower        : pd.Series — forecast - 1.96*residual_std (95% band lower)
        upper        : pd.Series — forecast + 1.96*residual_std (95% band upper)
        residual_std : float — std of historical residuals
    """
    n = len(monthly)
    t = np.arange(n, dtype=float)

    # 1. Linear trend
    coeffs     = np.polyfit(t, monthly.values, deg=1)
    trend_vals = np.polyval(coeffs, t)

    # 2. Seasonal indices (deviation from trend by calendar month)
    residuals   = monthly.values - trend_vals
    cal_months  = monthly.index.month
    seasonal_idx = {}
    for m in range(1, 13):
        mask = cal_months == m
        seasonal_idx[m] = float(np.mean(residuals[mask])) if mask.any() else 0.0

    # 3. Fitted values
    fitted = pd.Series(
        trend_vals + np.array([seasonal_idx[m] for m in cal_months]),
        index=monthly.index,
    )

    # 4. Residual std
    residual_std = float(np.std(monthly.values - fitted.values))

    # 5. Forecast
    future_t      = np.arange(n, n + periods_ahead, dtype=float)
    future_trend  = np.polyval(coeffs, future_t)
    last_date     = monthly.index[-1]
    future_dates  = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=periods_ahead,
        freq="MS",
    )
    future_months = future_dates.month
    future_seas   = np.array([seasonal_idx[m] for m in future_months])
    forecast_vals = future_trend + future_seas

    forecast = pd.Series(forecast_vals, index=future_dates)
    lower    = forecast - 1.96 * residual_std
    upper    = forecast + 1.96 * residual_std

    return trend_vals, seasonal_idx, fitted, forecast, lower, upper, residual_std


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

BG     = "#F8FAFC"
BORDER = "#334155"

def plot_forecast(monthly, fitted, forecast, lower, upper, save_path=None):
    """
    Plots historical revenue, trend fit, and 6-month forecast
    with a 95% confidence band.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Historical
    ax.plot(monthly.index, monthly.values,
            color="#94A3B8", lw=1.2, label="Actual revenue", zorder=2)
    ax.fill_between(monthly.index, monthly.values, alpha=0.08, color="#94A3B8")

    # Fitted
    ax.plot(fitted.index, fitted.values,
            color="#2563EB", lw=1.8, linestyle="--",
            label="Trend + seasonal fit", zorder=3)

    # Forecast
    ax.plot(forecast.index, forecast.values,
            color="#10B981", lw=2.2, marker="o", markersize=5,
            markerfacecolor="white", markeredgewidth=1.5,
            label="6-month forecast", zorder=4)

    # Confidence band
    ax.fill_between(forecast.index, lower.values, upper.values,
                    color="#10B981", alpha=0.15, label="95% confidence band")

    # Divider
    ax.axvline(monthly.index[-1], color="#CBD5E1", lw=1.2, linestyle=":")

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"PS{v/1000:.0f}K")
    )
    ax.set_title("Monthly Revenue — Historical Trend + 6-Month Forecast",
                 fontsize=13, fontweight="bold", color=BORDER, pad=10)
    ax.set_xlabel("Month", fontsize=9, color="#475569")
    ax.set_ylabel("Revenue", fontsize=9, color="#475569")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    ax.tick_params(colors="#475569", labelsize=8)
    ax.grid(axis="y", color="#E2E8F0", lw=0.7, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


def plot_seasonal_pattern(seasonal_idx, save_path=None):
    """Bar chart of seasonal indices — which months run above/below trend."""
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    values = [seasonal_idx[m] for m in range(1, 13)]
    colors = ["#10B981" if v >= 0 else "#EF4444" for v in values]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.bar(month_names, values, color=colors, edgecolor="white", width=0.65)
    ax.axhline(0, color=BORDER, lw=0.8)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"PS{v/1000:.0f}K")
    )
    ax.set_title("Seasonal Pattern — Monthly Deviation from Trend",
                 fontsize=12, fontweight="bold", color=BORDER, pad=8)
    ax.set_ylabel("Avg deviation from trend", fontsize=9, color="#475569")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    ax.tick_params(colors="#475569", labelsize=9)
    ax.grid(axis="y", color="#E2E8F0", lw=0.7, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_data.csv", parse_dates=["InvoiceDate"])
    monthly = prepare_monthly_revenue(df)

    trend_vals, seasonal_idx, fitted, forecast, lower, upper, res_std = \
        decompose_and_forecast(monthly, periods_ahead=6)

    print("=== 6-Month Revenue Forecast ===")
    result = pd.DataFrame({"forecast": forecast, "lower_95": lower, "upper_95": upper})
    print(result.to_string())
    print(f"\nResidual std: PS{res_std:,.0f}")
    print(f"Trend slope: PS{np.polyfit(range(len(monthly)), monthly.values, 1)[0]:,.0f} / month")

    plot_forecast(monthly, fitted, forecast, lower, upper)
    plot_seasonal_pattern(seasonal_idx)
    plt.show()
