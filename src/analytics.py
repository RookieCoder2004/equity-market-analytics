"""
High-Level Analytics Orchestrator.
Aggregates individual asset and portfolio performance/risk statistics
into formatted summary tables and structured reporting payloads.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.returns import (
    TRADING_DAYS_PER_YEAR,
    calculate_annualized_return,
    calculate_cumulative_returns,
    calculate_daily_returns,
    calculate_total_return,
)
from src.risk_metrics import (
    calculate_annualized_volatility,
    calculate_beta,
    calculate_correlation_matrix,
    calculate_covariance_matrix,
    calculate_cvar,
    calculate_downside_deviation,
    calculate_historical_var,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from src.portfolio import (
    calculate_diversification_ratio,
    calculate_portfolio_daily_returns,
    calculate_portfolio_metrics,
    calculate_risk_contribution,
    validate_weights,
)


def generate_asset_summary_table(
    prices_df: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.04,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> pd.DataFrame:
    """
    Generates institutional multi-asset summary tear sheet.
    """
    if prices_df.empty:
        return pd.DataFrame()

    returns_df = calculate_daily_returns(prices_df)
    tickers = list(prices_df.columns)

    latest_prices = prices_df.iloc[-1]
    total_ret = calculate_total_return(prices_df)
    cagr = calculate_annualized_return(returns_df, periods_per_year=periods_per_year)
    ann_vol = calculate_annualized_volatility(returns_df, periods_per_year=periods_per_year)
    sharpe = calculate_sharpe_ratio(returns_df, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
    sortino = calculate_sortino_ratio(returns_df, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
    mdd = calculate_max_drawdown(returns_df)
    var_95 = calculate_historical_var(returns_df, confidence_level=0.95)
    cvar_95 = calculate_cvar(returns_df, confidence_level=0.95)

    betas = {}
    for t in tickers:
        if benchmark_returns is not None and not benchmark_returns.empty:
            betas[t] = calculate_beta(returns_df[t], benchmark_returns)
        else:
            betas[t] = 1.0

    summary_df = pd.DataFrame({
        "Ticker": tickers,
        "Latest Price ($)": [latest_prices[t] for t in tickers],
        "Total Return": [total_ret[t] for t in tickers],
        "CAGR (Ann. Return)": [cagr[t] for t in tickers],
        "Ann. Volatility": [ann_vol[t] for t in tickers],
        "Sharpe Ratio": [sharpe[t] for t in tickers],
        "Sortino Ratio": [sortino[t] for t in tickers],
        "Max Drawdown": [mdd[t] for t in tickers],
        "Beta": [betas[t] for t in tickers],
        "Hist VaR (95%)": [var_95[t] for t in tickers],
        "CVaR (95%)": [cvar_95[t] for t in tickers]
    }).set_index("Ticker")

    return summary_df


def generate_comprehensive_risk_report(
    returns_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.04
) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Assembles a complete portfolio and risk analytics bundle.
    """
    report = {
        "correlation_matrix": calculate_correlation_matrix(returns_df),
        "covariance_matrix": calculate_covariance_matrix(returns_df, annualized=True),
    }

    if weights:
        valid_w = validate_weights(weights)
        report["portfolio_metrics"] = calculate_portfolio_metrics(
            returns_df, valid_w, benchmark_returns=benchmark_returns, risk_free_rate=risk_free_rate
        )
        report["risk_attribution"] = calculate_risk_contribution(returns_df, valid_w)
        report["diversification_ratio"] = calculate_diversification_ratio(returns_df, valid_w)

    return report
