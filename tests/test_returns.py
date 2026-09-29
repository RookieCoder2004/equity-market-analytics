"""Unit tests for returns analytics functions."""

import numpy as np
import pandas as pd
import pytest

from src.returns import (
    calculate_annualized_return,
    calculate_cumulative_returns,
    calculate_daily_returns,
    calculate_normalized_prices,
    calculate_rolling_returns,
    calculate_total_return,
)


@pytest.fixture
def sample_price_series():
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    return pd.Series([100.0, 105.0, 102.9, 108.045, 113.44725], index=dates, name="AAPL")


@pytest.fixture
def sample_price_df():
    dates = pd.date_range("2023-01-01", periods=4, freq="D")
    return pd.DataFrame({
        "AAPL": [100.0, 110.0, 104.5, 114.95],
        "MSFT": [200.0, 204.0, 208.08, 203.9184]
    }, index=dates)


def test_calculate_daily_returns(sample_price_series):
    returns = calculate_daily_returns(sample_price_series, method="simple")
    assert len(returns) == 4
    np.testing.assert_almost_equal(returns.iloc[0], 0.05, decimal=4)
    np.testing.assert_almost_equal(returns.iloc[1], -0.02, decimal=4)
    np.testing.assert_almost_equal(returns.iloc[2], 0.05, decimal=4)
    np.testing.assert_almost_equal(returns.iloc[3], 0.05, decimal=4)


def test_calculate_log_returns(sample_price_series):
    log_ret = calculate_daily_returns(sample_price_series, method="log")
    assert len(log_ret) == 4
    expected_log_0 = np.log(105.0 / 100.0)
    np.testing.assert_almost_equal(log_ret.iloc[0], expected_log_0, decimal=5)


def test_calculate_cumulative_returns(sample_price_series):
    daily = calculate_daily_returns(sample_price_series)
    cum_ret = calculate_cumulative_returns(daily)
    assert len(cum_ret) == 4
    # Final price is 113.44725 from 100.0, so cumulative return is 0.1344725
    np.testing.assert_almost_equal(cum_ret.iloc[-1], 0.1344725, decimal=5)


def test_calculate_normalized_prices(sample_price_df):
    norm = calculate_normalized_prices(sample_price_df, base_value=100.0)
    np.testing.assert_almost_equal(norm.iloc[0]["AAPL"], 100.0, decimal=5)
    np.testing.assert_almost_equal(norm.iloc[0]["MSFT"], 100.0, decimal=5)
    np.testing.assert_almost_equal(norm.iloc[1]["AAPL"], 110.0, decimal=5)
    np.testing.assert_almost_equal(norm.iloc[1]["MSFT"], 102.0, decimal=5)


def test_calculate_total_return(sample_price_series, sample_price_df):
    tot_s = calculate_total_return(sample_price_series)
    np.testing.assert_almost_equal(tot_s, 0.1344725, decimal=5)

    tot_df = calculate_total_return(sample_price_df)
    np.testing.assert_almost_equal(tot_df["AAPL"], 0.1495, decimal=4)
    np.testing.assert_almost_equal(tot_df["MSFT"], 0.019592, decimal=5)


def test_calculate_annualized_return_cagr():
    # 252 trading days with constant 0.1% daily return
    dates = pd.date_range("2023-01-01", periods=253, freq="B")
    daily_r = 0.001
    prices = pd.Series(100.0 * (1.0 + daily_r) ** np.arange(253), index=dates)
    daily_returns = calculate_daily_returns(prices)

    cagr = calculate_annualized_return(daily_returns, periods_per_year=252, geometric=True)
    expected_cagr = (1.0 + daily_r) ** 252 - 1.0
    np.testing.assert_almost_equal(cagr, expected_cagr, decimal=4)


def test_calculate_rolling_returns():
    ret = pd.Series([0.01, 0.02, -0.01, 0.03, 0.02])
    roll = calculate_rolling_returns(ret, window=3)
    assert len(roll) == 3
    # First rolling window for indices 0, 1, 2: (1.01 * 1.02 * 0.99) - 1
    expected_0 = (1.01 * 1.02 * 0.99) - 1.0
    np.testing.assert_almost_equal(roll.iloc[0], expected_0, decimal=5)
