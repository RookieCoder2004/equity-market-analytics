"""
Portfolio Construction & Risk Attribution Engine.
Handles weight validation, multi-asset portfolio returns, risk decomposition
(Marginal Risk Contribution & Percentage Contribution to Risk), diversification statistics,
portfolio comparison, and Markowitz Efficient Frontier generation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.returns import (
    TRADING_DAYS_PER_YEAR,
    calculate_annualized_return,
    calculate_cumulative_returns,
    calculate_total_return,
)
from src.risk_metrics import (
    calculate_annualized_volatility,
    calculate_beta,
    calculate_covariance_matrix,
    calculate_cvar,
    calculate_downside_deviation,
    calculate_historical_var,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)


def validate_weights(
    weights: Dict[str, float],
    allowed_tickers: Optional[List[str]] = None,
    tolerance: float = 1e-3
) -> Dict[str, float]:
    """
    Validates that:
    1. All weights are non-negative.
    2. Sum of weights equals 1.0 (within tolerance).
    3. Tickers are valid and formatted uppercase.

    Returns sanitized weight dictionary.
    """
    if not weights:
        raise ValueError("Weights dictionary cannot be empty.")

    sanitized = {}
    for ticker, w in weights.items():
        t_clean = ticker.strip().upper()
        if allowed_tickers and t_clean not in [t.upper() for t in allowed_tickers]:
            raise ValueError(f"Ticker '{t_clean}' is not in allowed asset universe.")
        if w < 0:
            raise ValueError(f"Weight for '{t_clean}' cannot be negative (got {w}).")
        sanitized[t_clean] = float(w)

    total_weight = sum(sanitized.values())
    if abs(total_weight - 1.0) > tolerance:
        raise ValueError(
            f"Portfolio weights must sum to 100% (got {total_weight * 100:.2f}%)."
        )

    # Re-normalize slightly to ensure exact 1.0 sum
    return {k: v / total_weight for k, v in sanitized.items()}


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Helper that forces positive weights to sum exactly to 1.0."""
    clean = {k.strip().upper(): max(0.0, float(v)) for k, v in weights.items()}
    total = sum(clean.values())
    if total <= 0:
        n = len(clean)
        return {k: 1.0 / n for k in clean}
    return {k: v / total for k, v in clean.items()}


def calculate_portfolio_daily_returns(
    returns_df: pd.DataFrame, weights: Dict[str, float]
) -> pd.Series:
    """
    Calculates weighted daily portfolio return series:
    R_p(t) = sum_{i=1}^N (w_i * r_{i,t})
    """
    valid_w = validate_weights(weights)
    tickers = list(valid_w.keys())

    missing = set(tickers) - set(returns_df.columns)
    if missing:
        raise ValueError(f"Returns DataFrame missing required tickers: {missing}")

    sub_returns = returns_df[tickers].dropna()
    weight_vector = np.array([valid_w[t] for t in tickers])

    portfolio_returns = sub_returns.dot(weight_vector)
    portfolio_returns.name = "Portfolio"
    return portfolio_returns


def calculate_portfolio_metrics(
    returns_df: pd.DataFrame,
    weights: Dict[str, float],
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.04,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> Dict[str, Any]:
    """
    Computes comprehensive risk and return analytics for a weighted portfolio.
    """
    valid_w = validate_weights(weights)
    port_ret = calculate_portfolio_daily_returns(returns_df, valid_w)

    ann_ret = calculate_annualized_return(port_ret, periods_per_year=periods_per_year)
    ann_vol = calculate_annualized_volatility(port_ret, periods_per_year=periods_per_year)
    sharpe = calculate_sharpe_ratio(port_ret, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
    sortino = calculate_sortino_ratio(port_ret, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
    max_dd = calculate_max_drawdown(port_ret)
    var_95 = calculate_historical_var(port_ret, confidence_level=0.95)
    var_99 = calculate_historical_var(port_ret, confidence_level=0.99)
    cvar_95 = calculate_cvar(port_ret, confidence_level=0.95)
    downside_dev = calculate_downside_deviation(port_ret, target_return=risk_free_rate, periods_per_year=periods_per_year)
    cum_ret = (1.0 + port_ret).prod() - 1.0

    beta = 1.0
    if benchmark_returns is not None and not benchmark_returns.empty:
        beta = calculate_beta(port_ret, benchmark_returns)

    div_ratio = calculate_diversification_ratio(returns_df, valid_w, periods_per_year=periods_per_year)

    return {
        "Total Return": float(cum_ret),
        "Annualized Return": float(ann_ret),
        "Annualized Volatility": float(ann_vol),
        "Sharpe Ratio": float(sharpe),
        "Sortino Ratio": float(sortino),
        "Maximum Drawdown": float(max_dd),
        "Beta (vs Benchmark)": float(beta),
        "Historical VaR (95%)": float(var_95),
        "Historical VaR (99%)": float(var_99),
        "Conditional VaR (95%)": float(cvar_95),
        "Downside Deviation": float(downside_dev),
        "Diversification Ratio": float(div_ratio),
        "Weights": valid_w
    }


def calculate_risk_contribution(
    returns_df: pd.DataFrame,
    weights: Dict[str, float],
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> pd.DataFrame:
    """
    Decomposes portfolio risk into:
    - Marginal Contribution to Risk (MCR): d(sigma_p)/d(w_i) = (Sigma * w)_i / sigma_p
    - Percentage Component Contribution to Risk (PCTR): w_i * MCR_i / sigma_p
    The sum of PCTR across all assets equals 100%.
    """
    valid_w = validate_weights(weights)
    tickers = list(valid_w.keys())
    sub_returns = returns_df[tickers].dropna()

    cov_matrix = calculate_covariance_matrix(sub_returns, annualized=True, periods_per_year=periods_per_year)
    w = np.array([valid_w[t] for t in tickers])

    port_var = float(w.T @ cov_matrix.values @ w)
    port_vol = np.sqrt(max(1e-8, port_var))

    # Marginal Risk Contribution
    mcr = (cov_matrix.values @ w) / port_vol

    # Absolute Risk Contribution = w_i * MCR_i
    abs_rc = w * mcr

    # Percentage Risk Contribution = (w_i * MCR_i) / port_vol * 100
    pct_rc = (abs_rc / port_vol) * 100.0

    # Individual asset volatilities
    indiv_vol = np.sqrt(np.diag(cov_matrix.values))

    df_risk = pd.DataFrame({
        "Ticker": tickers,
        "Weight": [valid_w[t] for t in tickers],
        "Asset Volatility": indiv_vol,
        "Marginal Risk Contribution (MCR)": mcr,
        "Absolute Risk Contribution": abs_rc,
        "Percentage Risk Contribution (PCTR %)": pct_rc
    })
    df_risk.sort_values(by="Percentage Risk Contribution (PCTR %)", ascending=False, inplace=True)
    return df_risk


def calculate_diversification_ratio(
    returns_df: pd.DataFrame,
    weights: Dict[str, float],
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Computes Diversification Ratio (Choueifaty & Coignard, 2008):
    DR = sum(w_i * sigma_i) / sigma_portfolio >= 1.0
    A higher diversification ratio indicates superior risk diversification.
    """
    valid_w = validate_weights(weights)
    tickers = list(valid_w.keys())
    sub_returns = returns_df[tickers].dropna()

    indiv_vol = sub_returns.std(ddof=1) * np.sqrt(periods_per_year)
    weighted_vol = sum(valid_w[t] * indiv_vol[t] for t in tickers)

    cov_matrix = calculate_covariance_matrix(sub_returns, annualized=True, periods_per_year=periods_per_year)
    w = np.array([valid_w[t] for t in tickers])
    port_vol = float(np.sqrt(max(1e-8, w.T @ cov_matrix.values @ w)))

    return float(weighted_vol / port_vol) if port_vol > 0 else 1.0


def compare_portfolios(
    returns_df: pd.DataFrame,
    portfolios: Dict[str, Dict[str, float]],
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.04
) -> pd.DataFrame:
    """
    Generates side-by-side performance & risk comparison across multiple allocations.
    """
    rows = []
    for name, w in portfolios.items():
        try:
            metrics = calculate_portfolio_metrics(
                returns_df, w, benchmark_returns=benchmark_returns, risk_free_rate=risk_free_rate
            )
            rows.append({
                "Portfolio": name,
                "Annualized Return": metrics["Annualized Return"],
                "Annualized Volatility": metrics["Annualized Volatility"],
                "Sharpe Ratio": metrics["Sharpe Ratio"],
                "Sortino Ratio": metrics["Sortino Ratio"],
                "Maximum Drawdown": metrics["Maximum Drawdown"],
                "Beta": metrics["Beta (vs Benchmark)"],
                "Hist VaR (95%)": metrics["Historical VaR (95%)"],
                "Conditional VaR (95%)": metrics["Conditional VaR (95%)"],
                "Diversification Ratio": metrics["Diversification Ratio"]
            })
        except Exception as e:
            continue

    if not rows:
        return pd.DataFrame()

    df_comp = pd.DataFrame(rows).set_index("Portfolio")
    return df_comp


def generate_efficient_frontier(
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.04,
    num_portfolios: int = 2000,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> Dict[str, Any]:
    """
    Calculates Markowitz Efficient Frontier via quadratic optimization & Monte Carlo simulation.
    Finds:
    - Maximum Sharpe Ratio Portfolio
    - Minimum Volatility Portfolio
    - Simulated portfolios cloud
    """
    tickers = list(returns_df.columns)
    n_assets = len(tickers)
    if n_assets < 2:
        raise ValueError("Need at least 2 assets to generate an efficient frontier.")

    clean_ret = returns_df.dropna()
    mean_daily = clean_ret.mean()
    cov_annual = calculate_covariance_matrix(clean_ret, annualized=True, periods_per_year=periods_per_year).values

    def port_performance(weights_arr):
        w = np.array(weights_arr)
        ret = float(np.sum(mean_daily * w) * periods_per_year)
        vol = float(np.sqrt(np.dot(w.T, np.dot(cov_annual, w))))
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
        return ret, vol, sharpe

    # 1. Optimize Max Sharpe Ratio
    def neg_sharpe(w):
        return -port_performance(w)[2]

    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    init_guess = np.repeat(1.0 / n_assets, n_assets)

    opt_sharpe = minimize(
        neg_sharpe, init_guess, method="SLSQP", bounds=bounds, constraints=constraints
    )
    max_sharpe_w = {tickers[i]: float(opt_sharpe.x[i]) for i in range(n_assets)}
    max_sharpe_ret, max_sharpe_vol, max_sharpe_sr = port_performance(opt_sharpe.x)

    # 2. Optimize Min Volatility
    def min_vol_func(w):
        return port_performance(w)[1]

    opt_vol = minimize(
        min_vol_func, init_guess, method="SLSQP", bounds=bounds, constraints=constraints
    )
    min_vol_w = {tickers[i]: float(opt_vol.x[i]) for i in range(n_assets)}
    min_vol_ret, min_vol_vol, min_vol_sr = port_performance(opt_vol.x)

    # 3. Monte Carlo Simulation
    sim_returns = []
    sim_volatilities = []
    sim_sharpes = []
    sim_weights = []

    np.random.seed(42)
    for _ in range(num_portfolios):
        w = np.random.random(n_assets)
        w /= np.sum(w)
        r, v, s = port_performance(w)
        sim_returns.append(r)
        sim_volatilities.append(v)
        sim_sharpes.append(s)
        sim_weights.append(w)

    return {
        "max_sharpe": {
            "weights": max_sharpe_w,
            "return": float(max_sharpe_ret),
            "volatility": float(max_sharpe_vol),
            "sharpe": float(max_sharpe_sr)
        },
        "min_volatility": {
            "weights": min_vol_w,
            "return": float(min_vol_ret),
            "volatility": float(min_vol_vol),
            "sharpe": float(min_vol_sr)
        },
        "simulation": {
            "returns": np.array(sim_returns),
            "volatilities": np.array(sim_volatilities),
            "sharpes": np.array(sim_sharpes),
            "weights": sim_weights
        },
        "tickers": tickers
    }
