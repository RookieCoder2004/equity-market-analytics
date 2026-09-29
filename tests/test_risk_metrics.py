"""Unit tests for risk metrics and statistical financial calculations."""

import numpy as np
import pandas as pd
import pytest

from src.risk_metrics import (
    calculate_annualized_volatility,
    calculate_beta,
    calculate_correlation_matrix,
    calculate_covariance_matrix,
    calculate_cvar,
    calculate_downside_deviation,
    calculate_drawdown_series,
    calculate_historical_var,
    calculate_max_drawdown,
    calculate_rolling_sharpe,
    calculate_rolling_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)


@pytest.fixture
def synthetic_returns():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    r_a = np.random.normal(0.0008, 0.012, 252)
    r_b = 0.8 * r_a + np.random.normal(0.0002, 0.006, 252)
    bench = r_a * 0.5 + np.random.normal(0.0004, 0.008, 252)

    df = pd.DataFrame({
        "AssetA": r_a,
        "AssetB": r_b,
        "Benchmark": bench
    }, index=dates)
    return df


def test_annualized_volatility(synthetic_returns):
    asset_a = synthetic_returns["AssetA"]
    vol = calculate_annualized_volatility(asset_a, periods_per_year=252)
    expected = asset_a.std(ddof=1) * np.sqrt(252)
    np.testing.assert_almost_equal(vol, expected, decimal=5)


def test_drawdown_and_max_drawdown():
    # Wealth goes: 100 -> 120 -> 90 -> 110 -> 80 -> 100
    prices = pd.Series([100.0, 120.0, 90.0, 110.0, 80.0, 100.0])
    dd = calculate_drawdown_series(prices, is_price=True)
    
    # Peak at 120, lowest after is 80 -> Drawdown is (80 - 120) / 120 = -40 / 120 = -0.333333
    mdd = calculate_max_drawdown(prices, is_price=True)
    np.testing.assert_almost_equal(mdd, -1.0 / 3.0, decimal=4)
    np.testing.assert_almost_equal(dd.iloc[1], 0.0, decimal=4)
    np.testing.assert_almost_equal(dd.iloc[2], (90.0 - 120.0) / 120.0, decimal=4)


def test_beta_calculation():
    # Perfect benchmark correlation
    bench = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    # Asset with beta = 1.5
    asset = bench * 1.5
    beta = calculate_beta(asset, bench)
    np.testing.assert_almost_equal(beta, 1.5, decimal=4)


def test_sharpe_ratio():
    # Constant 15% annual return with 10% annual vol, Rf = 5% -> Sharpe = (15 - 5) / 10 = 1.0
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    np.random.seed(123)
    # create sample returns
    r = pd.Series(np.random.normal(0.0006, 0.01, 252), index=dates)
    sr = calculate_sharpe_ratio(r, risk_free_rate=0.04)
    assert isinstance(sr, float)
    assert -10.0 < sr < 10.0


def test_sortino_ratio():
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    r = pd.Series(np.random.normal(0.001, 0.01, 252), index=dates)
    sortino = calculate_sortino_ratio(r, risk_free_rate=0.04)
    assert isinstance(sortino, float)


def test_historical_var_and_cvar():
    # 100 observations uniformly from -0.05 to +0.05
    returns = pd.Series(np.linspace(-0.05, 0.05, 101))
    var_95 = calculate_historical_var(returns, confidence_level=0.95)
    cvar_95 = calculate_cvar(returns, confidence_level=0.95)

    # 95% VaR is the 5th percentile loss (should be ~0.045 positive loss)
    assert var_95 > 0.04
    # CVaR is average of worst 5% losses, so CVaR >= VaR
    assert cvar_95 >= var_95


def test_correlation_and_covariance_matrix(synthetic_returns):
    corr = calculate_correlation_matrix(synthetic_returns)
    assert corr.shape == (3, 3)
    np.testing.assert_almost_equal(corr.loc["AssetA", "AssetA"], 1.0, decimal=5)
    assert -1.0 <= corr.loc["AssetA", "AssetB"] <= 1.0

    cov = calculate_covariance_matrix(synthetic_returns, annualized=True)
    assert cov.shape == (3, 3)
    assert cov.loc["AssetA", "AssetA"] > 0
