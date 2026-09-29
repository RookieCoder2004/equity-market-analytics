"""
Financial Visualization Suite.
Produces institutional-grade, publication-quality Matplotlib & Seaborn charts
with standardized financial typography, color palettes, and formatting.
"""

from typing import Any, Dict, List, Optional, Union
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.returns import (
    calculate_cumulative_returns,
    calculate_daily_returns,
    calculate_normalized_prices,
)
from src.risk_metrics import (
    calculate_drawdown_series,
    calculate_rolling_volatility,
)

# Apply sleek styling defaults
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]


def plot_normalized_prices(
    prices_df: pd.DataFrame,
    base_value: float = 100.0,
    title: str = "Normalized Price Performance (Base = 100)"
) -> plt.Figure:
    """Plots normalized price trajectories starting from base_value."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    norm_df = calculate_normalized_prices(prices_df, base_value=base_value)

    for i, col in enumerate(norm_df.columns):
        ax.plot(norm_df.index, norm_df[col], label=col, linewidth=1.8, color=PALETTE[i % len(PALETTE)])

    ax.axhline(base_value, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=10, labelpad=8)
    ax.set_ylabel(f"Indexed Value (Base={base_value:.0f})", fontsize=10, labelpad=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()
    return fig


def plot_cumulative_returns(
    returns_df: Union[pd.Series, pd.DataFrame],
    title: str = "Cumulative Returns Over Time"
) -> plt.Figure:
    """Plots cumulative compound growth curves."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    cum_ret = calculate_cumulative_returns(returns_df)

    if isinstance(cum_ret, pd.Series):
        ax.plot(cum_ret.index, cum_ret.values, label=cum_ret.name or "Asset", linewidth=2.0, color="#1f77b4")
        ax.fill_between(cum_ret.index, 0, cum_ret.values, alpha=0.15, color="#1f77b4")
    else:
        for i, col in enumerate(cum_ret.columns):
            ax.plot(cum_ret.index, cum_ret[col], label=col, linewidth=1.8, color=PALETTE[i % len(PALETTE)])

    ax.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.6)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=10, labelpad=8)
    ax.set_ylabel("Cumulative Return", fontsize=10, labelpad=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()
    return fig


def plot_return_distributions(
    returns_df: Union[pd.Series, pd.DataFrame],
    ticker: Optional[str] = None,
    bins: int = 40
) -> plt.Figure:
    """Plots return distribution histogram with KDE and normal curve comparison."""
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=120)

    if isinstance(returns_df, pd.DataFrame):
        target_col = ticker or returns_df.columns[0]
        data = returns_df[target_col].dropna()
    else:
        target_col = returns_df.name or "Asset"
        data = returns_df.dropna()

    mu, std = data.mean(), data.std()

    # Histogram & KDE
    sns.histplot(data, bins=bins, kde=True, stat="density", color="#1f77b4", alpha=0.4, ax=ax, label="Empirical KDE")

    # Fitted Normal Curve
    x = np.linspace(data.min(), data.max(), 200)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(x, p, "r--", linewidth=1.6, label=f"Normal Fit (μ={mu*100:.2f}%, σ={std*100:.2f}%)")

    # Value at Risk 95% line
    var_95 = np.percentile(data, 5)
    ax.axvline(var_95, color="#d62728", linestyle=":", linewidth=2, label=f"95% Daily VaR ({-var_95*100:.2f}%)")

    ax.set_title(f"Daily Return Distribution: {target_col}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Daily Return", fontsize=10, labelpad=8)
    ax.set_ylabel("Probability Density", fontsize=10, labelpad=8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8.5)
    plt.tight_layout()
    return fig


def plot_rolling_volatility(
    returns_df: Union[pd.Series, pd.DataFrame],
    window: int = 30,
    title: Optional[str] = None
) -> plt.Figure:
    """Plots rolling annualized standard deviation."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    roll_vol = calculate_rolling_volatility(returns_df, window=window)

    if isinstance(roll_vol, pd.Series):
        ax.plot(roll_vol.index, roll_vol.values, label=roll_vol.name or "Asset", linewidth=1.8, color="#ff7f0e")
    else:
        for i, col in enumerate(roll_vol.columns):
            ax.plot(roll_vol.index, roll_vol[col], label=col, linewidth=1.8, color=PALETTE[i % len(PALETTE)])

    plot_title = title or f"Rolling Annualized Volatility ({window}-Day Window)"
    ax.set_title(plot_title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=10, labelpad=8)
    ax.set_ylabel("Annualized Volatility", fontsize=10, labelpad=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()
    return fig


def plot_drawdowns(
    returns_df: Union[pd.Series, pd.DataFrame],
    title: str = "Historical Underwater Drawdown"
) -> plt.Figure:
    """Plots peak-to-trough drawdowns with shaded underwater area."""
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=120)
    dd_df = calculate_drawdown_series(returns_df)

    if isinstance(dd_df, pd.Series):
        ax.plot(dd_df.index, dd_df.values, label=dd_df.name or "Drawdown", linewidth=1.5, color="#d62728")
        ax.fill_between(dd_df.index, 0, dd_df.values, alpha=0.3, color="#d62728")
    else:
        for i, col in enumerate(dd_df.columns):
            ax.plot(dd_df.index, dd_df[col], label=col, linewidth=1.5, color=PALETTE[i % len(PALETTE)])

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=10, labelpad=8)
    ax.set_ylabel("Drawdown from Peak", fontsize=10, labelpad=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylim([min(-0.05, float(np.min(dd_df.values)) * 1.1), 0.02])
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    returns_df: pd.DataFrame,
    title: str = "Asset Return Correlation Matrix"
) -> plt.Figure:
    """Renders correlation matrix heatmap with values annotated."""
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=120)
    corr = returns_df.corr(method="pearson")

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        ax=ax
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_portfolio_allocation(
    weights: Dict[str, float],
    title: str = "Portfolio Asset Allocation"
) -> plt.Figure:
    """Generates a professional donut chart of portfolio weights."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=120)

    # Filter out near-zero weights
    clean_w = {k: v for k, v in weights.items() if v > 0.001}
    labels = [f"{k}\n({v*100:.1f}%)" for k, v in clean_w.items()]
    values = list(clean_w.values())

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        pctdistance=0.75,
        startangle=90,
        colors=PALETTE[:len(values)],
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )

    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight("bold")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_risk_return_scatter(
    asset_summary_df: pd.DataFrame,
    portfolio_metric: Optional[Dict[str, float]] = None,
    risk_free_rate: float = 0.04
) -> plt.Figure:
    """Plots risk vs return scatter with Capital Allocation reference."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)

    x = asset_summary_df["Ann. Volatility"]
    y = asset_summary_df["CAGR (Ann. Return)"]
    labels = asset_summary_df.index

    # Scatter points for assets
    ax.scatter(x, y, color="#1f77b4", s=120, zorder=5, edgecolors="black", linewidth=1.2, label="Individual Assets")

    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (x.iloc[i], y.iloc[i]),
            textcoords="offset points",
            xytext=(6, 6),
            ha="left",
            fontweight="bold",
            fontsize=9
        )

    # Portfolio point if provided
    if portfolio_metric:
        p_vol = portfolio_metric["Annualized Volatility"]
        p_ret = portfolio_metric["Annualized Return"]
        ax.scatter(
            [p_vol], [p_ret],
            color="#d62728", s=220, marker="*", zorder=6,
            edgecolors="black", linewidth=1.2, label="Constructed Portfolio"
        )
        ax.annotate(
            "Portfolio",
            (p_vol, p_ret),
            textcoords="offset points",
            xytext=(8, 8),
            fontweight="bold",
            color="#d62728",
            fontsize=10
        )

    ax.axhline(risk_free_rate, color="green", linestyle="--", linewidth=1, label=f"Risk-Free Rate ({risk_free_rate*100:.1f}%)")
    ax.set_title("Risk vs Return (Annualized Volatility vs CAGR)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Annualized Volatility (Risk)", fontsize=10, labelpad=8)
    ax.set_ylabel("Annualized Return (CAGR)", fontsize=10, labelpad=8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()
    return fig


def plot_portfolio_comparison(
    comparison_df: pd.DataFrame,
    metrics_to_plot: Optional[List[str]] = None
) -> plt.Figure:
    """Bar chart comparing multiple portfolio configurations."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)

    cols = metrics_to_plot or ["Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Sortino Ratio"]
    available = [c for c in cols if c in comparison_df.columns]

    df_subset = comparison_df[available]
    df_subset.plot(kind="bar", ax=ax, colormap="viridis", width=0.7, edgecolor="black", linewidth=0.8)

    ax.set_title("Portfolio Allocation Performance & Risk Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Portfolio Allocation", fontsize=10, labelpad=8)
    ax.set_ylabel("Metric Value", fontsize=10, labelpad=8)
    ax.set_xticklabels(comparison_df.index, rotation=0, fontweight="bold")
    ax.legend(frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()
    return fig


def plot_efficient_frontier(
    frontier_data: Dict[str, Any],
    risk_free_rate: float = 0.04
) -> plt.Figure:
    """Plots Markowitz Efficient Frontier with simulated cloud and optimal portfolios."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    sim = frontier_data["simulation"]
    vols = sim["volatilities"]
    rets = sim["returns"]
    sharpes = sim["sharpes"]

    # Monte carlo scatter cloud
    scatter = ax.scatter(
        vols, rets, c=sharpes, cmap="plasma", marker="o", s=15, alpha=0.6,
        label="Simulated Allocations"
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio", fontsize=10)

    # Max Sharpe Portfolio
    ms = frontier_data["max_sharpe"]
    ax.scatter(
        [ms["volatility"]], [ms["return"]],
        color="#2ca02c", marker="*", s=280, edgecolors="black", linewidth=1.5,
        label=f"Max Sharpe (SR={ms['sharpe']:.2f}, Ret={ms['return']*100:.1f}%)", zorder=6
    )

    # Min Volatility Portfolio
    mv = frontier_data["min_volatility"]
    ax.scatter(
        [mv["volatility"]], [mv["return"]],
        color="#1f77b4", marker="D", s=150, edgecolors="black", linewidth=1.5,
        label=f"Min Volatility (Vol={mv['volatility']*100:.1f}%, Ret={mv['return']*100:.1f}%)", zorder=6
    )

    # Capital Allocation Line (CAL) from Rf through Max Sharpe
    cal_x = np.linspace(0, max(vols) * 1.1, 50)
    cal_y = risk_free_rate + ((ms["return"] - risk_free_rate) / ms["volatility"]) * cal_x
    ax.plot(cal_x, cal_y, "g--", linewidth=1.5, alpha=0.8, label="Capital Allocation Line (CAL)")

    ax.set_title("Markowitz Efficient Frontier & Capital Allocation Line", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Annualized Volatility (Risk)", fontsize=10, labelpad=8)
    ax.set_ylabel("Annualized Return (CAGR)", fontsize=10, labelpad=8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlim([0, max(vols) * 1.15])
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()
    return fig


def plot_rolling_sharpe(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    window: int = 63,
    risk_free_rate: float = 0.04
) -> plt.Figure:
    """Plots rolling Sharpe ratio comparison over time."""
    from src.risk_metrics import calculate_rolling_sharpe

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=120)
    port_sr = calculate_rolling_sharpe(portfolio_returns, window=window, risk_free_rate=risk_free_rate)

    ax.plot(port_sr.index, port_sr.values, label=f"Portfolio ({window}-Day Rolling Sharpe)", color="#1f77b4", linewidth=1.8)

    if benchmark_returns is not None and not benchmark_returns.empty:
        bench_sr = calculate_rolling_sharpe(benchmark_returns, window=window, risk_free_rate=risk_free_rate)
        ax.plot(bench_sr.index, bench_sr.values, label=f"Benchmark ({window}-Day Rolling Sharpe)", color="#7f7f7f", linestyle="--", linewidth=1.5)

    ax.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.axhline(1.0, color="green", linestyle=":", linewidth=0.8, alpha=0.5, label="Good Risk-Adjusted (SR=1.0)")

    ax.set_title(f"Rolling Sharpe Ratio Trend ({window}-Day Window)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=10, labelpad=8)
    ax.set_ylabel("Sharpe Ratio", fontsize=10, labelpad=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()
    return fig
