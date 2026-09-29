# Equity Market Analytics & Portfolio Risk Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24%2B-FF4B4B.svg)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/Database-SQLite3-003B57.svg)](https://www.sqlite.org/)
[![Pytest](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](https://pytest.org/)

An institutional-grade quantitative finance analytics and portfolio risk platform built in Python. The platform enables multi-asset historical data ingestion, relational SQL persistence, quantitative return and risk modeling, risk attribution (Marginal & Percentage Contribution to Risk), Markowitz Efficient Frontier optimization, and real-time portfolio stress testing through an interactive Streamlit dashboard.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Quantitative Finance Metrics & Formulas](#quantitative-finance-metrics--formulas)
5. [Tech Stack](#tech-stack)
6. [Project Structure](#project-structure)
7. [Installation & Setup](#installation--setup)
8. [Running the Application](#running-the-application)
9. [Running Automated Tests](#running-automated-tests)
10. [Example Portfolio Configuration](#example-portfolio-configuration)
11. [Future Roadmap](#future-roadmap)

---

## Project Overview

In quantitative portfolio management, evaluating asset allocation strategies requires moving beyond raw price returns. Portfolio managers need deep insights into risk-adjusted performance, downside risk, historical tail events (Value at Risk / Expected Shortfall), factor sensitivity (Beta), and risk decomposition (identifying which assets drive portfolio volatility).

This platform delivers an end-to-end quantitative workflow:
- **Data Ingestion**: Robust automated pipeline collecting daily OHLCV equity and benchmark data.
- **Relational Persistence**: SQLite database schema with indexes for fast time-series querying.
- **Risk & Attribution Engine**: Modular financial mathematics for Sharpe, Sortino, Drawdowns, Beta, Historical VaR/CVaR, and Marginal Risk Contributions.
- **Portfolio Construction & Optimization**: Markowitz Mean-Variance optimization, Monte Carlo frontier simulation, and multi-strategy comparisons.
- **Interactive UI**: Multi-tab Streamlit dashboard with institutional visualizations.

---

## Key Features

- 🌐 **Automated Data Ingestion & SQL Caching**: Fetches historical data for any universe of equities (`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `NVDA`, `TSLA`, `JPM`, `META`, etc.) and benchmarks (`^GSPC`, `SPY`, `QQQ`). Includes graceful fallback simulation when offline.
- 💾 **Relational SQL Database (SQLite)**: Tables for stock metadata, historical OHLCV prices, saved portfolio allocations, and analytics results.
- 📈 **Return & Performance Analytics**: Simple returns, log returns, cumulative growth, CAGR, rolling returns, and normalized price trajectories.
- 🛡️ **Comprehensive Risk Analytics**:
  - Annualized Volatility & Rolling Volatility
  - Peak-to-Trough Underwater Drawdowns & Maximum Drawdown (MDD)
  - CAPM Beta against benchmark indices
  - Sharpe Ratio & Sortino Ratio (with downside semi-deviation)
  - 1-Day Historical Value at Risk (VaR 95% & 99%) and Conditional VaR (CVaR / Expected Shortfall)
- 💼 **Portfolio Builder & Risk Attribution**:
  - Interactive weight adjustment with real-time validation (weights non-negative and summing to 100%).
  - Marginal Contribution to Risk ($MCR$) and Percentage Contribution to Risk ($PCTR$).
  - Diversification Ratio (Choueifaty & Coignard).
- ⚖️ **Strategy Comparison & Efficient Frontier**:
  - Compare custom allocations against 60/40 Equity/Defensive and Equal-Weight strategies.
  - Markowitz Mean-Variance quadratic optimization calculating Maximum Sharpe and Minimum Volatility portfolios.
- 🎨 **Institutional Visualization Suite**: Publication-quality Matplotlib/Seaborn charts with proper financial formatting.

---

## System Architecture

```
                               ┌─────────────────────────────┐
                               │   Market Data Providers     │
                               │    (yfinance / Fallback)    │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │   Data Loader & Cleaning    │
                               │   (src/data_loader.py)      │
                               └──────────────┬──────────────┘
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       │                                             │
                       ▼                                             ▼
        ┌─────────────────────────────┐               ┌─────────────────────────────┐
        │       SQLite Database       │               │      Analytics Engines      │
        │    (data/market_data.db)    │◄─────────────►│ - src/returns.py             │
        │ - stocks                    │               │ - src/risk_metrics.py       │
        │ - historical_prices         │               │ - src/portfolio.py          │
        │ - portfolio_allocations     │               │ - src/analytics.py          │
        │ - analytics_results         │               └──────────────┬──────────────┘
        └─────────────────────────────┘                              │
                                                                     ▼
                                                      ┌─────────────────────────────┐
                                                      │    Visualization Suite      │
                                                      │   (src/visualization.py)    │
                                                      └──────────────┬──────────────┘
                                                                     │
                                                                     ▼
                                                      ┌─────────────────────────────┐
                                                      │    Streamlit Dashboard      │
                                                      │         (app.py)            │
                                                      └─────────────────────────────┘
```

---

## Quantitative Finance Metrics & Formulas

All calculations assume $N = 252$ trading days per calendar year.

### 1. Daily Simple Return
$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

### 2. Compound Annual Growth Rate (CAGR)
$$\text{CAGR} = \left( \prod_{t=1}^T (1 + r_t) \right)^{\frac{252}{T}} - 1$$

### 3. Annualized Volatility
$$\sigma_{\text{ann}} = \sigma_{\text{daily}} \times \sqrt{252}$$

### 4. Maximum Drawdown (MDD)
$$\text{Wealth}(t) = \prod_{s=1}^t (1 + r_s), \quad \text{Peak}(t) = \max_{s \le t} \text{Wealth}(s)$$
$$\text{Drawdown}(t) = \frac{\text{Wealth}(t) - \text{Peak}(t)}{\text{Peak}(t)}$$
$$\text{MDD} = \min_{t} \text{Drawdown}(t)$$

### 5. CAPM Beta
$$\beta = \frac{\text{Cov}(R_i, R_m)}{\text{Var}(R_m)}$$

### 6. Sharpe Ratio
$$\text{Sharpe Ratio} = \frac{R_{\text{ann}} - R_f}{\sigma_{\text{ann}}}$$

### 7. Downside Semi-Deviation & Sortino Ratio
$$\sigma_d = \sqrt{ \frac{1}{T} \sum_{t=1}^T \left( \min(r_t - r_{\text{target}}, 0) \right)^2 } \times \sqrt{252}$$
$$\text{Sortino Ratio} = \frac{R_{\text{ann}} - R_f}{\sigma_d}$$

### 8. Historical Value at Risk (VaR) & CVaR (Expected Shortfall)
$$\text{VaR}_{\alpha} = - \text{Percentile}(R_{\text{daily}}, 1 - \alpha)$$
$$\text{CVaR}_{\alpha} = - \mathbb{E}[R \mid R \le -\text{VaR}_{\alpha}]$$

### 9. Portfolio Variance & Volatility
$$\sigma_p = \sqrt{ \mathbf{w}^T \mathbf{\Sigma} \mathbf{w} }$$

### 10. Marginal & Percentage Contribution to Risk (PCTR)
$$\text{MCR}_i = \frac{(\mathbf{\Sigma} \mathbf{w})_i}{\sigma_p}$$
$$\text{PCTR}_i = \frac{w_i \times \text{MCR}_i}{\sigma_p} \times 100\%, \quad \text{where } \sum_{i=1}^n \text{PCTR}_i = 100\%$$

### 11. Diversification Ratio
$$\text{DR} = \frac{\sum_{i=1}^n w_i \sigma_i}{\sigma_p} \ge 1.0$$

---

## Tech Stack

- **Core**: Python 3.9+
- **Data Manipulation & Mathematics**: `pandas`, `numpy`, `scipy`
- **Database**: SQLite3 (relational schema with compound indexes)
- **Market Data Feed**: `yfinance` with intelligent rate-limiting & fallback generator
- **Visualizations**: `matplotlib`, `seaborn`
- **Interactive UI**: `streamlit`
- **Testing**: `pytest`

---

## Project Structure

```
equity-market-analytics/
├── app.py                      # Main interactive Streamlit application
├── requirements.txt            # Pinned dependencies
├── README.md                   # Comprehensive platform documentation
├── .gitignore                  # Git ignore rules
│
├── data/
│   └── market_data.db          # SQLite database storing raw prices, securities & results
│
├── src/
│   ├── __init__.py
│   ├── database.py             # SQLite schema, queries, persistence engine
│   ├── data_loader.py          # Data ingestion (yfinance + fallback simulation)
│   ├── returns.py              # Daily, cumulative, CAGR, and rolling return math
│   ├── risk_metrics.py         # Volatility, Sharpe, Sortino, Drawdown, Beta, VaR, CVaR
│   ├── portfolio.py            # Portfolio metrics, risk attribution, comparison, frontier
│   ├── analytics.py            # Summary orchestrator and reporting tear sheets
│   └── visualization.py        # Institutional-grade Matplotlib/Seaborn visualization suite
│
├── tests/
│   ├── __init__.py
│   ├── test_returns.py         # Unit tests for return calculations
│   ├── test_risk_metrics.py    # Unit tests for risk metrics and tail statistics
│   └── test_portfolio.py       # Unit tests for portfolio optimization and risk decomposition
│
└── notebooks/
    └── exploratory_analysis.ipynb # End-to-end quantitative research notebook
```

---

## Installation & Setup

1. **Clone the repository or navigate to the project directory**:
   ```bash
   cd equity-market-analytics
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Dashboard Navigation:
- **🌐 Market Overview**: Select tickers and date range; view top metric cards and $100 growth comparison.
- **📊 Stock Analysis**: Inspect individual asset performance, rolling volatility, drawdowns, and empirical return distributions with fitted Gaussian and 95% VaR markers.
- **🔗 Correlation & Risk**: View Pearson correlation heatmaps, covariance matrices, and risk/Beta summary tables.
- **💼 Portfolio Builder**: Custom weight assignment with real-time validation, donut allocation chart, and benchmark backtest.
- **🛡️ Portfolio Risk**: Risk attribution table (Marginal & Percentage Contribution to Risk), 95%/99% VaR, 95% CVaR, and rolling Sharpe ratio.
- **⚖️ Portfolio Comparison & Frontier**: Side-by-side comparison with 60/40 and Equal-Weight strategies, accompanied by Markowitz Efficient Frontier optimization.

---

## Running Automated Tests

The platform includes comprehensive unit tests verifying return math, risk metrics, portfolio weight validation, and risk attribution:

```bash
pytest -v tests/
```

Expected Output:
```
tests/test_portfolio.py::test_validate_weights_success PASSED
tests/test_portfolio.py::test_validate_weights_negative PASSED
tests/test_portfolio.py::test_validate_weights_sum_error PASSED
tests/test_portfolio.py::test_normalize_weights PASSED
tests/test_portfolio.py::test_portfolio_daily_returns PASSED
tests/test_portfolio.py::test_portfolio_metrics PASSED
tests/test_portfolio.py::test_risk_contribution_sums_to_100 PASSED
tests/test_portfolio.py::test_diversification_ratio PASSED
tests/test_portfolio.py::test_compare_portfolios PASSED
tests/test_portfolio.py::test_generate_efficient_frontier PASSED
tests/test_returns.py::test_calculate_daily_returns PASSED
tests/test_returns.py::test_calculate_log_returns PASSED
tests/test_returns.py::test_calculate_cumulative_returns PASSED
tests/test_returns.py::test_calculate_normalized_prices PASSED
tests/test_returns.py::test_calculate_total_return PASSED
tests/test_returns.py::test_calculate_annualized_return_cagr PASSED
tests/test_returns.py::test_calculate_rolling_returns PASSED
tests/test_risk_metrics.py::test_annualized_volatility PASSED
tests/test_risk_metrics.py::test_drawdown_and_max_drawdown PASSED
tests/test_risk_metrics.py::test_beta_calculation PASSED
tests/test_risk_metrics.py::test_sharpe_ratio PASSED
tests/test_risk_metrics.py::test_sortino_ratio PASSED
tests/test_risk_metrics.py::test_historical_var_and_cvar PASSED
tests/test_risk_metrics.py::test_correlation_and_covariance_matrix PASSED

============================== 24 passed in 0.61s ==============================
```

---

## Example Portfolio Configuration

Consider the following sample diversified tech/financial allocation:

| Ticker | Company Name | Target Weight |
|---|---|---|
| **AAPL** | Apple Inc. | 25.0% |
| **MSFT** | Microsoft Corp. | 20.0% |
| **NVDA** | NVIDIA Corp. | 20.0% |
| **AMZN** | Amazon.com Inc. | 15.0% |
| **GOOGL**| Alphabet Inc. | 10.0% |
| **JPM**  | JPMorgan Chase & Co. | 10.0% |

### Sample Analysis Results:
- **Annualized Return (CAGR)**: ~24.5%
- **Annualized Volatility**: ~21.2%
- **Sharpe Ratio ($R_f = 4\%$)**: ~0.97
- **Sortino Ratio**: ~1.38
- **Maximum Drawdown**: -26.4%
- **1-Day 95% Historical VaR**: 1.95%
- **Diversification Ratio**: ~1.18x

---

## Future Roadmap

- [ ] Black-Litterman model integration for incorporating investor views into prior equilibrium returns.
- [ ] Factor attribution (Fama-French 3-factor and 5-factor regression analysis).
- [ ] GARCH(1,1) volatility forecasting module.
- [ ] Support for options Greeks and non-linear derivative risk profiles.
- [ ] Automated daily portfolio rebalancing backtester with transaction cost modeling.
