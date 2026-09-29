"""
Returns Analytics Engine.
Calculates simple daily returns, log returns, cumulative returns,
annualized returns (CAGR / Arithmetic), rolling returns, and normalized price trajectories.
"""

from typing import Union
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def calculate_daily_returns(
    prices: Union[pd.Series, pd.DataFrame], method: str = "simple"
) -> Union[pd.Series, pd.DataFrame]:
    """
    Computes daily percentage returns from adjusted close price series or DataFrame.

    Parameters:
    -----------
    prices : pd.Series or pd.DataFrame
        Historical price series or price matrix.
    method : str
        'simple' for (P_t - P_{t-1}) / P_{t-1} or 'log' for ln(P_t / P_{t-1}).

    Returns:
    --------
    pd.Series or pd.DataFrame of daily returns with initial NaN dropped.
    """
    if prices.empty:
        return prices.copy()

    if method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()

    return returns.dropna(how="all")


def calculate_cumulative_returns(
    returns: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    """
    Computes cumulative compound returns from daily simple returns:
    R_cum(t) = Product_{i=1}^t (1 + r_i) - 1

    Parameters:
    -----------
    returns : pd.Series or pd.DataFrame
        Daily simple returns.

    Returns:
    --------
    pd.Series or pd.DataFrame of cumulative returns.
    """
    if returns.empty:
        return returns.copy()

    return (1.0 + returns).cumprod() - 1.0


def calculate_normalized_prices(
    prices: Union[pd.Series, pd.DataFrame], base_value: float = 100.0
) -> Union[pd.Series, pd.DataFrame]:
    """
    Normalizes price series to start at a common base value (e.g. 100).
    """
    if prices.empty:
        return prices.copy()

    first_valid = prices.iloc[0]
    return (prices / first_valid) * base_value


def calculate_total_return(prices: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Calculates total holding period return:
    R_total = (P_end - P_start) / P_start
    """
    if prices.empty:
        return 0.0

    if isinstance(prices, pd.Series):
        clean = prices.dropna()
        if len(clean) < 2:
            return 0.0
        return float((clean.iloc[-1] - clean.iloc[0]) / clean.iloc[0])
    else:
        clean = prices.dropna(how="all")
        if len(clean) < 2:
            return pd.Series(0.0, index=prices.columns)
        return (clean.iloc[-1] - clean.iloc[0]) / clean.iloc[0]


def calculate_annualized_return(
    returns_or_prices: Union[pd.Series, pd.DataFrame],
    is_price: bool = False,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    geometric: bool = True
) -> Union[float, pd.Series]:
    """
    Calculates annualized return (CAGR or arithmetic annualized).

    Formulas:
    ---------
    CAGR = (1 + R_total) ^ (periods_per_year / N) - 1
    Arithmetic = mean(r_daily) * periods_per_year
    """
    if returns_or_prices.empty:
        return 0.0 if isinstance(returns_or_prices, pd.Series) else pd.Series(0.0, index=returns_or_prices.columns)

    if is_price:
        returns = calculate_daily_returns(returns_or_prices)
    else:
        returns = returns_or_prices

    if isinstance(returns, pd.Series):
        clean_ret = returns.dropna()
        n_periods = len(clean_ret)
        if n_periods < 2:
            return 0.0

        if geometric:
            compounded = float((1.0 + clean_ret).prod())
            if compounded <= 0:
                return -1.0
            return float(compounded ** (periods_per_year / n_periods) - 1.0)
        else:
            return float(clean_ret.mean() * periods_per_year)
    else:
        result = {}
        for col in returns.columns:
            clean_col = returns[col].dropna()
            n = len(clean_col)
            if n < 2:
                result[col] = 0.0
            elif geometric:
                prod = float((1.0 + clean_col).prod())
                result[col] = float(prod ** (periods_per_year / n) - 1.0) if prod > 0 else -1.0
            else:
                result[col] = float(clean_col.mean() * periods_per_year)
        return pd.Series(result)


def calculate_rolling_returns(
    returns: Union[pd.Series, pd.DataFrame],
    window: int = 21,
    annualized: bool = False,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculates rolling window compound returns.
    """
    if returns.empty:
        return returns.copy()

    # Rolling compound return = product of (1 + r) over window - 1
    rolling_compound = (1.0 + returns).rolling(window=window).apply(np.prod, raw=True) - 1.0

    if annualized:
        rolling_compound = (1.0 + rolling_compound) ** (periods_per_year / window) - 1.0

    return rolling_compound.dropna(how="all")
