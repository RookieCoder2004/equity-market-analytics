"""Unit tests for portfolio optimization, weights validation, and risk attribution."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import (
    calculate_diversification_ratio,
    calculate_portfolio_daily_returns,
    calculate_portfolio_metrics,
    calculate_risk_contribution,
    compare_portfolios,
    generate_efficient_frontier,
    normalize_weights,
    validate_weights,
)


@pytest.fixture
def sample_returns_df():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    r1 = np.random.normal(0.0008, 0.015, 252)
    r2 = np.random.normal(0.0006, 0.012, 252)
    r3 = np.random.normal(0.0004, 0.008, 252)
    return pd.DataFrame({
        "AAPL": r1,
        "MSFT": r2,
        "JPM": r3
    }, index=dates)


def test_validate_weights_success():
    weights = {"AAPL": 0.5, "MSFT": 0.3, "JPM": 0.2}
    validated = validate_weights(weights, allowed_tickers=["AAPL", "MSFT", "JPM"])
    assert sum(validated.values()) == pytest.approx(1.0, 1e-6)
    assert validated["AAPL"] == 0.5


def test_validate_weights_negative():
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_weights({"AAPL": -0.1, "MSFT": 1.1})


def test_validate_weights_sum_error():
    with pytest.raises(ValueError, match="must sum to 100%"):
        validate_weights({"AAPL": 0.4, "MSFT": 0.4})


def test_normalize_weights():
    w = {"AAPL": 20, "MSFT": 80}
    norm_w = normalize_weights(w)
    assert norm_w["AAPL"] == pytest.approx(0.2)
    assert norm_w["MSFT"] == pytest.approx(0.8)


def test_portfolio_daily_returns(sample_returns_df):
    weights = {"AAPL": 0.5, "MSFT": 0.5, "JPM": 0.0}
    port_ret = calculate_portfolio_daily_returns(sample_returns_df, weights)
    expected = 0.5 * sample_returns_df["AAPL"] + 0.5 * sample_returns_df["MSFT"]
    pd.testing.assert_series_equal(port_ret, expected, check_names=False)


def test_portfolio_metrics(sample_returns_df):
    weights = {"AAPL": 0.4, "MSFT": 0.4, "JPM": 0.2}
    metrics = calculate_portfolio_metrics(sample_returns_df, weights, risk_free_rate=0.04)
    assert "Annualized Return" in metrics
    assert "Annualized Volatility" in metrics
    assert "Sharpe Ratio" in metrics
    assert "Historical VaR (95%)" in metrics
    assert metrics["Annualized Volatility"] > 0


def test_risk_contribution_sums_to_100(sample_returns_df):
    weights = {"AAPL": 0.5, "MSFT": 0.3, "JPM": 0.2}
    rc_df = calculate_risk_contribution(sample_returns_df, weights)
    assert "Percentage Risk Contribution (PCTR %)" in rc_df.columns
    total_pct_rc = rc_df["Percentage Risk Contribution (PCTR %)"].sum()
    np.testing.assert_almost_equal(total_pct_rc, 100.0, decimal=2)


def test_diversification_ratio(sample_returns_df):
    weights = {"AAPL": 0.33, "MSFT": 0.33, "JPM": 0.34}
    dr = calculate_diversification_ratio(sample_returns_df, weights)
    # Diversification ratio for non-identical imperfectly correlated assets should be >= 1.0
    assert dr >= 0.99


def test_compare_portfolios(sample_returns_df):
    portfolios = {
        "Balanced": {"AAPL": 0.33, "MSFT": 0.33, "JPM": 0.34},
        "Tech-Heavy": {"AAPL": 0.70, "MSFT": 0.30, "JPM": 0.0}
    }
    comp_df = compare_portfolios(sample_returns_df, portfolios)
    assert len(comp_df) == 2
    assert "Annualized Return" in comp_df.columns
    assert "Sharpe Ratio" in comp_df.columns


def test_generate_efficient_frontier(sample_returns_df):
    frontier = generate_efficient_frontier(sample_returns_df, num_portfolios=200)
    assert "max_sharpe" in frontier
    assert "min_volatility" in frontier
    assert "simulation" in frontier

    # Weights for Max Sharpe must sum to 1
    ms_weights = frontier["max_sharpe"]["weights"]
    assert sum(ms_weights.values()) == pytest.approx(1.0, 1e-4)

    # Volatility of Min Vol portfolio must be <= Volatility of Max Sharpe portfolio
    assert frontier["min_volatility"]["volatility"] <= frontier["max_sharpe"]["volatility"] + 1e-4
