"""
Risk Analytics Module.
Calculates statistical and financial risk metrics including annualized volatility,
rolling volatility, peak-to-trough drawdowns, benchmark Beta, correlation/covariance matrices,
Sharpe ratio, Sortino ratio, downside deviation, Historical VaR & CVaR, and rolling Sharpe.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.returns import (
    TRADING_DAYS_PER_YEAR,
    calculate_annualized_return,
    calculate_cumulative_returns,
)


def calculate_annualized_volatility(
    returns: Union[pd.Series, pd.DataFrame], periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> Union[float, pd.Series]:
    """
    Computes annualized standard deviation of returns:
    sigma_ann = sigma_daily * sqrt(periods_per_year)
    """
    if returns.empty:
        return 0.0 if isinstance(returns, pd.Series) else pd.Series(0.0, index=returns.columns)

    if isinstance(returns, pd.Series):
        clean = returns.dropna()
        if len(clean) < 2:
            return 0.0
        return float(clean.std(ddof=1) * np.sqrt(periods_per_year))
    else:
        return returns.std(ddof=1) * np.sqrt(periods_per_year)


def calculate_rolling_volatility(
    returns: Union[pd.Series, pd.DataFrame],
    window: int = 30,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> Union[pd.Series, pd.DataFrame]:
    """
    Computes rolling annualized volatility over a sliding window.
    """
    if returns.empty:
        return returns.copy()

    return (returns.rolling(window=window).std(ddof=1) * np.sqrt(periods_per_year)).dropna(how="all")


def calculate_drawdown_series(
    returns_or_prices: Union[pd.Series, pd.DataFrame], is_price: bool = False
) -> Union[pd.Series, pd.DataFrame]:
    """
    Computes underwater drawdown series:
    Drawdown(t) = (Wealth(t) - Peak(t)) / Peak(t)
    """
    if returns_or_prices.empty:
        return returns_or_prices.copy()

    if is_price:
        wealth = returns_or_prices
    else:
        wealth = (1.0 + returns_or_prices).cumprod()

    if isinstance(wealth, pd.Series):
        peaks = wealth.cummax()
        drawdown = (wealth - peaks) / peaks
        return drawdown
    else:
        peaks = wealth.cummax()
        drawdown = (wealth - peaks) / peaks
        return drawdown


def calculate_max_drawdown(
    returns_or_prices: Union[pd.Series, pd.DataFrame], is_price: bool = False
) -> Union[float, pd.Series]:
    """
    Calculates Maximum Drawdown (MDD) as a negative fraction.
    """
    dd = calculate_drawdown_series(returns_or_prices, is_price=is_price)
    if dd.empty:
        return 0.0 if isinstance(returns_or_prices, pd.Series) else pd.Series(0.0, index=returns_or_prices.columns)

    if isinstance(dd, pd.Series):
        return float(dd.min())
    else:
        return dd.min()


def calculate_beta(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculates CAPM Beta against a benchmark index:
    Beta = Cov(R_asset, R_bench) / Var(R_bench)
    """
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 5:
        return 1.0

    asset_col = aligned.iloc[:, 0]
    bench_col = aligned.iloc[:, 1]

    cov = np.cov(asset_col, bench_col)[0, 1]
    var_bench = np.var(bench_col, ddof=1)

    if var_bench == 0:
        return 1.0

    return float(cov / var_bench)


def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes pairwise Pearson correlation matrix across asset returns.
    """
    if returns_df.empty:
        return pd.DataFrame()
    return returns_df.corr(method="pearson")


def calculate_covariance_matrix(
    returns_df: pd.DataFrame,
    annualized: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> pd.DataFrame:
    """
    Computes covariance matrix (optionally annualized).
    """
    if returns_df.empty:
        return pd.DataFrame()

    cov = returns_df.cov()
    if annualized:
        cov = cov * periods_per_year
    return cov


def calculate_sharpe_ratio(
    returns: Union[pd.Series, pd.DataFrame],
    risk_free_rate: float = 0.04,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> Union[float, pd.Series]:
    """
    Calculates Annualized Sharpe Ratio:
    Sharpe = (Ann_Return - Rf) / Ann_Volatility
    """
    ann_ret = calculate_annualized_return(returns, periods_per_year=periods_per_year)
    ann_vol = calculate_annualized_volatility(returns, periods_per_year=periods_per_year)

    if isinstance(returns, pd.Series):
        if ann_vol == 0:
            return 0.0
        return float((ann_ret - risk_free_rate) / ann_vol)
    else:
        vol_clean = ann_vol.replace(0, np.nan)
        return ((ann_ret - risk_free_rate) / vol_clean).fillna(0.0)


def calculate_downside_deviation(
    returns: Union[pd.Series, pd.DataFrame],
    target_return: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> Union[float, pd.Series]:
    """
    Calculates annualized downside deviation (semi-standard deviation below target):
    Downside_Dev = sqrt( mean( min(r_t - target_daily, 0)^2 ) ) * sqrt(periods_per_year)
    """
    if returns.empty:
        return 0.0 if isinstance(returns, pd.Series) else pd.Series(0.0, index=returns.columns)

    target_daily = target_return / periods_per_year

    if isinstance(returns, pd.Series):
        clean = returns.dropna()
        if len(clean) == 0:
            return 0.0
        diff = clean - target_daily
        downside = diff[diff < 0]
        if len(downside) == 0:
            return 0.0
        semi_var = np.mean(downside ** 2)
        return float(np.sqrt(semi_var) * np.sqrt(periods_per_year))
    else:
        result = {}
        for col in returns.columns:
            clean = returns[col].dropna()
            if len(clean) == 0:
                result[col] = 0.0
                continue
            diff = clean - target_daily
            downside = diff[diff < 0]
            if len(downside) == 0:
                result[col] = 0.0
            else:
                semi_var = np.mean(downside ** 2)
                result[col] = float(np.sqrt(semi_var) * np.sqrt(periods_per_year))
        return pd.Series(result)


def calculate_sortino_ratio(
    returns: Union[pd.Series, pd.DataFrame],
    risk_free_rate: float = 0.04,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> Union[float, pd.Series]:
    """
    Calculates Sortino Ratio using downside deviation:
    Sortino = (Ann_Return - Rf) / Downside_Deviation
    """
    ann_ret = calculate_annualized_return(returns, periods_per_year=periods_per_year)
    downside_dev = calculate_downside_deviation(
        returns, target_return=risk_free_rate, periods_per_year=periods_per_year
    )

    if isinstance(returns, pd.Series):
        if downside_dev == 0:
            return 0.0
        return float((ann_ret - risk_free_rate) / downside_dev)
    else:
        dd_clean = downside_dev.replace(0, np.nan)
        return ((ann_ret - risk_free_rate) / dd_clean).fillna(0.0)


def calculate_historical_var(
    returns: Union[pd.Series, pd.DataFrame],
    confidence_level: float = 0.95
) -> Union[float, pd.Series]:
    """
    Calculates Historical Value at Risk (VaR) over a 1-day horizon.
    Returns a positive percentage value (representing maximum expected loss at (1 - alpha)).
    For example, 95% VaR of 0.025 means there is a 5% chance of losing more than 2.5% in a single day.
    """
    if returns.empty:
        return 0.0 if isinstance(returns, pd.Series) else pd.Series(0.0, index=returns.columns)

    quantile_level = 1.0 - confidence_level

    if isinstance(returns, pd.Series):
        clean = returns.dropna()
        if len(clean) == 0:
            return 0.0
        var_value = -float(np.percentile(clean, quantile_level * 100))
        return max(0.0, var_value)
    else:
        res = {}
        for col in returns.columns:
            clean = returns[col].dropna()
            if len(clean) == 0:
                res[col] = 0.0
            else:
                var_val = -float(np.percentile(clean, quantile_level * 100))
                res[col] = max(0.0, var_val)
        return pd.Series(res)


def calculate_cvar(
    returns: Union[pd.Series, pd.DataFrame],
    confidence_level: float = 0.95
) -> Union[float, pd.Series]:
    """
    Calculates Conditional Value at Risk (CVaR) / Expected Shortfall:
    CVaR = - Expected Value of returns that fall below the VaR threshold.
    """
    if returns.empty:
        return 0.0 if isinstance(returns, pd.Series) else pd.Series(0.0, index=returns.columns)

    quantile_level = 1.0 - confidence_level

    if isinstance(returns, pd.Series):
        clean = returns.dropna()
        if len(clean) == 0:
            return 0.0
        threshold = np.percentile(clean, quantile_level * 100)
        tail_losses = clean[clean <= threshold]
        if len(tail_losses) == 0:
            return 0.0
        return max(0.0, -float(tail_losses.mean()))
    else:
        res = {}
        for col in returns.columns:
            clean = returns[col].dropna()
            if len(clean) == 0:
                res[col] = 0.0
                continue
            threshold = np.percentile(clean, quantile_level * 100)
            tail_losses = clean[clean <= threshold]
            res[col] = max(0.0, -float(tail_losses.mean())) if len(tail_losses) > 0 else 0.0
        return pd.Series(res)


def calculate_rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
    risk_free_rate: float = 0.04,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> pd.Series:
    """
    Calculates rolling window Sharpe ratio.
    """
    if returns.empty or len(returns) < window:
        return pd.Series(dtype=float)

    rf_daily = risk_free_rate / periods_per_year
    excess_returns = returns - rf_daily

    rolling_mean = excess_returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std(ddof=1)

    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(periods_per_year)
    return rolling_sharpe.dropna()
