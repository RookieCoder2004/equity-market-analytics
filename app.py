"""
Equity Market Analytics & Portfolio Risk Analysis Platform
Interactive Streamlit Dashboard.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.analytics import (
    generate_asset_summary_table,
    generate_comprehensive_risk_report,
)
from src.data_loader import DEFAULT_BENCHMARK, DEFAULT_TICKERS, MarketDataLoader
from src.database import DatabaseManager
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
from src.returns import (
    calculate_annualized_return,
    calculate_cumulative_returns,
    calculate_daily_returns,
    calculate_normalized_prices,
    calculate_rolling_returns,
)
from src.risk_metrics import (
    calculate_annualized_volatility,
    calculate_beta,
    calculate_correlation_matrix,
    calculate_covariance_matrix,
    calculate_cvar,
    calculate_historical_var,
    calculate_max_drawdown,
    calculate_rolling_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from src.visualization import (
    plot_correlation_heatmap,
    plot_cumulative_returns,
    plot_drawdowns,
    plot_efficient_frontier,
    plot_normalized_prices,
    plot_portfolio_allocation,
    plot_portfolio_comparison,
    plot_return_distributions,
    plot_risk_return_scatter,
    plot_rolling_sharpe,
    plot_rolling_volatility,
)

# Page configuration
st.set_page_config(
    page_title="Equity Market Analytics & Portfolio Risk",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Database and Data Loader
@st.cache_resource
def get_backend():
    db = DatabaseManager("data/market_data.db")
    loader = MarketDataLoader(db)
    return db, loader

db_manager, data_loader = get_backend()

# Custom CSS styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a1a1a;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-weight: 600;
        border-radius: 6px 6px 0px 0px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.title("📊 Platform Controls")
st.sidebar.markdown("Configure market data parameters and benchmark settings.")

universe_options = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "JPM", "META", "BRK-B", "V", "JNJ", "WMT", "PG"]
selected_tickers = st.sidebar.multiselect(
    "Select Equity Universe",
    options=universe_options,
    default=DEFAULT_TICKERS
)

# Custom ticker write-in
custom_ticker_input = st.sidebar.text_input("Add Custom Ticker (e.g. NFLX, AMD):", "")
if custom_ticker_input:
    clean_custom = [t.strip().upper() for t in custom_ticker_input.split(",") if t.strip()]
    for t in clean_custom:
        if t not in selected_tickers:
            selected_tickers.append(t)

benchmark_ticker = st.sidebar.selectbox(
    "Benchmark Index",
    options=["^GSPC", "SPY", "QQQ", "^IXIC", "^DJI"],
    index=0
)

# Date range
col_d1, col_d2 = st.sidebar.columns(2)
default_start = datetime.today() - timedelta(days=365 * 3)
start_date = col_d1.date_input("Start Date", default_start)
end_date = col_d2.date_input("End Date", datetime.today())

# Risk-free rate
risk_free_rate_pct = st.sidebar.slider(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=4.0,
    step=0.25
)
risk_free_rate = risk_free_rate_pct / 100.0

refresh_data = st.sidebar.button("🔄 Sync & Refresh Market Data", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Equity Market Analytics Platform v1.0 • Institutional Quantitative Finance Suite")

# ----------------- DATA INGESTION -----------------
@st.cache_data(show_spinner=False)
def load_market_dataset(tickers, bench, start_d, end_d, force_refresh=False):
    all_symbols = list(set(tickers + [bench]))
    df = data_loader.fetch_and_store_ticker_data(
        tickers=all_symbols,
        start_date=start_d.strftime("%Y-%m-%d"),
        end_date=end_d.strftime("%Y-%m-%d"),
        use_cache=not force_refresh
    )
    price_matrix = db_manager.get_price_matrix(
        tickers=all_symbols,
        start_date=start_d.strftime("%Y-%m-%d"),
        end_date=end_d.strftime("%Y-%m-%d")
    )
    return price_matrix

if len(selected_tickers) == 0:
    st.warning("⚠️ Please select at least one ticker from the sidebar.")
    st.stop()

with st.spinner("Fetching and processing market data..."):
    try:
        raw_price_matrix = load_market_dataset(
            selected_tickers, benchmark_ticker, start_date, end_date, force_refresh=refresh_data
        )
    except Exception as e:
        st.error(f"Error loading market data: {e}")
        st.stop()

if raw_price_matrix.empty or len(raw_price_matrix) < 5:
    st.error("Insufficient market data returned for the selected date range. Please widen the date filter.")
    st.stop()

# Split universe and benchmark
valid_tickers = [t for t in selected_tickers if t in raw_price_matrix.columns]
if not valid_tickers:
    st.error("None of the selected tickers have price data.")
    st.stop()

prices_df = raw_price_matrix[valid_tickers].ffill().bfill().dropna()
returns_df = calculate_daily_returns(prices_df)

benchmark_returns = None
if benchmark_ticker in raw_price_matrix.columns:
    bench_prices = raw_price_matrix[benchmark_ticker].ffill().bfill().dropna()
    benchmark_returns = calculate_daily_returns(bench_prices)

# ----------------- MAIN NAVIGATION TABS -----------------
st.title("📈 Equity Market Analytics & Portfolio Risk Analysis")
st.markdown(
    f"Analyzing **{len(valid_tickers)} securities** from `{prices_df.index[0].strftime('%b %d, %Y')}` "
    f"to `{prices_df.index[-1].strftime('%b %d, %Y')}` (Benchmark: `{benchmark_ticker}`, $R_f = {risk_free_rate_pct:.1f}\\%$)."
)

tabs = st.tabs([
    "🌐 Market Overview",
    "📊 Stock Analysis",
    "🔗 Correlation & Risk",
    "💼 Portfolio Builder",
    "🛡️ Portfolio Risk",
    "⚖️ Portfolio Comparison & Frontier"
])

# ----------------- TAB A: MARKET OVERVIEW -----------------
with tabs[0]:
    st.subheader("Market Universe Performance Summary")

    summary_df = generate_asset_summary_table(
        prices_df,
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free_rate
    )

    # Top KPI Cards
    best_cagr_ticker = summary_df["CAGR (Ann. Return)"].idxmax()
    best_cagr_val = summary_df.loc[best_cagr_ticker, "CAGR (Ann. Return)"]
    lowest_vol_ticker = summary_df["Ann. Volatility"].idxmin()
    lowest_vol_val = summary_df.loc[lowest_vol_ticker, "Ann. Volatility"]
    best_sharpe_ticker = summary_df["Sharpe Ratio"].idxmax()
    best_sharpe_val = summary_df.loc[best_sharpe_ticker, "Sharpe Ratio"]

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.metric("Total Securities", len(valid_tickers))
    with kpi_col2:
        st.metric(f"Top Performer (CAGR)", f"{best_cagr_ticker} ({best_cagr_val*100:.1f}%)")
    with kpi_col3:
        st.metric(f"Lowest Volatility", f"{lowest_vol_ticker} ({lowest_vol_val*100:.1f}%)")
    with kpi_col4:
        st.metric(f"Highest Sharpe", f"{best_sharpe_ticker} ({best_sharpe_val:.2f})")

    st.markdown("### Institutional Summary Table")
    # Formatted display table
    display_summary = summary_df.copy()
    display_summary["Latest Price ($)"] = display_summary["Latest Price ($)"].apply(lambda x: f"${x:,.2f}")
    display_summary["Total Return"] = display_summary["Total Return"].apply(lambda x: f"{x*100:+.2f}%")
    display_summary["CAGR (Ann. Return)"] = display_summary["CAGR (Ann. Return)"].apply(lambda x: f"{x*100:+.2f}%")
    display_summary["Ann. Volatility"] = display_summary["Ann. Volatility"].apply(lambda x: f"{x*100:.2f}%")
    display_summary["Sharpe Ratio"] = display_summary["Sharpe Ratio"].apply(lambda x: f"{x:.2f}")
    display_summary["Sortino Ratio"] = display_summary["Sortino Ratio"].apply(lambda x: f"{x:.2f}")
    display_summary["Max Drawdown"] = display_summary["Max Drawdown"].apply(lambda x: f"{x*100:.2f}%")
    display_summary["Beta"] = display_summary["Beta"].apply(lambda x: f"{x:.2f}")
    display_summary["Hist VaR (95%)"] = display_summary["Hist VaR (95%)"].apply(lambda x: f"{x*100:.2f}%")
    display_summary["CVaR (95%)"] = display_summary["CVaR (95%)"].apply(lambda x: f"{x*100:.2f}%")

    st.dataframe(display_summary, use_container_width=True)

    # Normalized Growth Chart
    st.markdown("### Historical Growth of $100")
    fig_norm = plot_normalized_prices(prices_df, base_value=100.0)
    st.pyplot(fig_norm)
    plt.close(fig_norm)

# ----------------- TAB B: STOCK ANALYSIS -----------------
with tabs[1]:
    st.subheader("Deep-Dive Single Asset Diagnostics")
    selected_stock = st.selectbox("Select Security to Inspect", options=valid_tickers, index=0)

    stock_prices = prices_df[selected_stock]
    stock_returns = returns_df[selected_stock]

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    stock_cagr = calculate_annualized_return(stock_returns)
    stock_vol = calculate_annualized_volatility(stock_returns)
    stock_sr = calculate_sharpe_ratio(stock_returns, risk_free_rate=risk_free_rate)
    stock_mdd = calculate_max_drawdown(stock_returns)

    col_s1.metric(f"{selected_stock} Price", f"${stock_prices.iloc[-1]:.2f}")
    col_s2.metric("Annualized CAGR", f"{stock_cagr*100:+.2f}%")
    col_s3.metric("Annualized Volatility", f"{stock_vol*100:.2f}%")
    col_s4.metric("Max Drawdown", f"{stock_mdd*100:.2f}%")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Cumulative Compound Return")
        fig_cum = plot_cumulative_returns(stock_returns, title=f"{selected_stock} Cumulative Return")
        st.pyplot(fig_cum)
        plt.close(fig_cum)

    with c2:
        st.markdown("#### Rolling Volatility Dynamics")
        vol_win = st.slider("Rolling Window (Days)", min_value=10, max_value=126, value=30, key="stock_vol_win")
        fig_rvol = plot_rolling_volatility(stock_returns, window=vol_win, title=f"{selected_stock} Rolling Volatility ({vol_win}d)")
        st.pyplot(fig_rvol)
        plt.close(fig_rvol)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Historical Underwater Drawdown")
        fig_dd = plot_drawdowns(stock_returns, title=f"{selected_stock} Peak-to-Trough Drawdowns")
        st.pyplot(fig_dd)
        plt.close(fig_dd)

    with c4:
        st.markdown("#### Empirical Return Distribution & 95% VaR")
        fig_dist = plot_return_distributions(stock_returns, ticker=selected_stock)
        st.pyplot(fig_dist)
        plt.close(fig_dist)

# ----------------- TAB C: CORRELATION & RISK -----------------
with tabs[2]:
    st.subheader("Cross-Asset Correlation & Risk Decomposition")

    c_corr, c_cov = st.columns([1.2, 1])
    with c_corr:
        st.markdown("#### Asset Return Correlation Heatmap")
        fig_corr = plot_correlation_heatmap(returns_df)
        st.pyplot(fig_corr)
        plt.close(fig_corr)

    with c_cov:
        st.markdown("#### Annualized Covariance Matrix ($\\Sigma$)")
        cov_matrix = calculate_covariance_matrix(returns_df, annualized=True)
        st.dataframe(cov_matrix.style.format("{:.4f}"), use_container_width=True)

        st.markdown("#### Risk & CAPM Beta Metrics")
        risk_table = summary_df[["Ann. Volatility", "Beta", "Sharpe Ratio", "Hist VaR (95%)", "CVaR (95%)"]]
        st.dataframe(risk_table.style.format({
            "Ann. Volatility": "{:.2%}",
            "Beta": "{:.2f}",
            "Sharpe Ratio": "{:.2f}",
            "Hist VaR (95%)": "{:.2%}",
            "CVaR (95%)": "{:.2%}"
        }), use_container_width=True)

    st.markdown("### Risk vs Return Landscape")
    fig_scatter = plot_risk_return_scatter(summary_df, risk_free_rate=risk_free_rate)
    st.pyplot(fig_scatter)
    plt.close(fig_scatter)

# ----------------- TAB D: PORTFOLIO BUILDER -----------------
with tabs[3]:
    st.subheader("Custom Multi-Asset Portfolio Constructor")
    st.markdown("Select holdings and assign percentage allocations. Weights must sum to 100%.")

    port_assets = st.multiselect(
        "Select Portfolio Assets",
        options=valid_tickers,
        default=valid_tickers[:min(5, len(valid_tickers))]
    )

    if not port_assets:
        st.warning("Please select at least 1 asset for the portfolio.")
        st.stop()

    n_p = len(port_assets)
    st.markdown("#### Assign Asset Weights")

    eq_col1, eq_col2 = st.columns(2)
    equalize = eq_col1.button("⚖️ Set Equal Weights", use_container_width=True)
    normalize_btn = eq_col2.button("🎯 Auto-Normalize Current Weights to 100%", use_container_width=True)

    # User weight inputs in columns
    input_weights = {}
    cols = st.columns(min(n_p, 4))
    for i, asset in enumerate(port_assets):
        with cols[i % min(n_p, 4)]:
            default_val = 100.0 / n_p
            key_name = f"weight_slider_{asset}"
            if equalize or key_name not in st.session_state:
                st.session_state[key_name] = round(default_val, 1)

            val = st.number_input(
                f"{asset} Weight (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state[key_name]),
                step=1.0,
                key=key_name
            )
            input_weights[asset] = val / 100.0

    if normalize_btn:
        norm_w = normalize_weights(input_weights)
        for k, v in norm_w.items():
            st.session_state[f"weight_slider_{k}"] = round(v * 100.0, 1)
        st.rerun()

    total_w_pct = sum(input_weights.values()) * 100.0
    is_valid_weights = abs(total_w_pct - 100.0) <= 0.1

    if is_valid_weights:
        st.success(f"✅ Total Weight: {total_w_pct:.1f}% (Valid Allocation)")
        active_weights = validate_weights(input_weights)
    else:
        st.error(f"❌ Total Weight: {total_w_pct:.1f}% (Must equal 100.0%)")
        active_weights = normalize_weights(input_weights)
        st.info("Using normalized weights for preview below.")

    # Save portfolio to DB option
    p_save_col1, p_save_col2 = st.columns([3, 1])
    p_name_input = p_save_col1.text_input("Save Portfolio Preset Name:", "My Custom Strategy")
    if p_save_col2.button("💾 Save Allocation"):
        db_manager.save_portfolio_allocation(p_name_input, active_weights)
        st.success(f"Saved allocation '{p_name_input}' to SQLite database!")

    p_col1, p_col2 = st.columns([1, 1.4])
    with p_col1:
        st.markdown("#### Target Allocation Breakdown")
        fig_pie = plot_portfolio_allocation(active_weights)
        st.pyplot(fig_pie)
        plt.close(fig_pie)

    with p_col2:
        st.markdown("#### Backtest Performance vs Benchmark")
        port_daily = calculate_portfolio_daily_returns(returns_df[port_assets], active_weights)
        comp_df = pd.DataFrame({"Constructed Portfolio": port_daily})
        if benchmark_returns is not None:
            comp_df[f"Benchmark ({benchmark_ticker})"] = benchmark_returns

        fig_port_cum = plot_cumulative_returns(comp_df, title="Portfolio Growth vs Benchmark")
        st.pyplot(fig_port_cum)
        plt.close(fig_port_cum)

# ----------------- TAB E: PORTFOLIO RISK -----------------
with tabs[4]:
    st.subheader("🛡️ Deep Risk Attribution & Factor Analytics")

    p_metrics = calculate_portfolio_metrics(
        returns_df[port_assets],
        active_weights,
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free_rate
    )

    # 6 Top Metric Badges
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    m_col1.metric("CAGR Return", f"{p_metrics['Annualized Return']*100:+.2f}%")
    m_col2.metric("Annual Volatility", f"{p_metrics['Annualized Volatility']*100:.2f}%")
    m_col3.metric("Sharpe Ratio", f"{p_metrics['Sharpe Ratio']:.2f}")
    m_col4.metric("Sortino Ratio", f"{p_metrics['Sortino Ratio']:.2f}")
    m_col5.metric("Max Drawdown", f"{p_metrics['Maximum Drawdown']*100:.2f}%")
    m_col6.metric("Diversification Ratio", f"{p_metrics['Diversification Ratio']:.2f}x")

    st.markdown("---")
    r_col1, r_col2 = st.columns(2)

    with r_col1:
        st.markdown("### Risk Decomposition: Marginal & Percentage Contribution (PCTR)")
        st.caption("Deconstructs total portfolio volatility into each constituent asset's percentage contribution to overall risk.")
        risk_contrib_df = calculate_risk_contribution(returns_df[port_assets], active_weights)
        st.dataframe(
            risk_contrib_df.style.format({
                "Weight": "{:.1%}",
                "Asset Volatility": "{:.2%}",
                "Marginal Risk Contribution (MCR)": "{:.4f}",
                "Absolute Risk Contribution": "{:.4f}",
                "Percentage Risk Contribution (PCTR %)": "{:.2f}%"
            }),
            use_container_width=True
        )

    with r_col2:
        st.markdown("### Downside Risk & Value at Risk (VaR)")
        var_summary_df = pd.DataFrame({
            "Risk Metric": [
                "1-Day Historical VaR (95% Confidence)",
                "1-Day Historical VaR (99% Confidence)",
                "1-Day Conditional VaR / Expected Shortfall (95%)",
                "Annualized Downside Deviation",
                "Portfolio Beta vs Benchmark"
            ],
            "Value": [
                f"{p_metrics['Historical VaR (95%)']*100:.2f}%",
                f"{p_metrics['Historical VaR (99%)']*100:.2f}%",
                f"{p_metrics['Conditional VaR (95%)']*100:.2f}%",
                f"{p_metrics['Downside Deviation']*100:.2f}%",
                f"{p_metrics['Beta (vs Benchmark)']:.2f}"
            ]
        }).set_index("Risk Metric")
        st.table(var_summary_df)

    st.markdown("### Rolling Risk-Adjusted Quality (63-Day Sharpe Ratio)")
    port_ret_series = calculate_portfolio_daily_returns(returns_df[port_assets], active_weights)
    fig_roll_sr = plot_rolling_sharpe(port_ret_series, benchmark_returns=benchmark_returns, risk_free_rate=risk_free_rate)
    st.pyplot(fig_roll_sr)
    plt.close(fig_roll_sr)

# ----------------- TAB F: PORTFOLIO COMPARISON & FRONTIER -----------------
with tabs[5]:
    st.subheader("⚖️ Strategy Comparison & Markowitz Efficient Frontier")

    # Construct Predefined Strategies
    equal_weights = {t: 1.0 / len(valid_tickers) for t in valid_tickers}

    # 60/40 Equities / Defensive proxy (Defensive using lower vol assets or JPM/BRK/dividend)
    defensive_tickers = [t for t in ["JPM", "BRK-B", "JNJ", "WMT", "PG"] if t in valid_tickers]
    growth_tickers = [t for t in ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META"] if t in valid_tickers]

    strategy_portfolios = {
        "User Custom Portfolio": active_weights,
        "Equal-Weighted Universe": equal_weights,
    }

    if defensive_tickers and growth_tickers:
        w_60_40 = {}
        for g in growth_tickers:
            w_60_40[g] = 0.60 / len(growth_tickers)
        for d in defensive_tickers:
            w_60_40[d] = 0.40 / len(defensive_tickers)
        strategy_portfolios["60/40 Growth/Defensive"] = normalize_weights(w_60_40)

    comparison_df = compare_portfolios(
        returns_df,
        strategy_portfolios,
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free_rate
    )

    st.markdown("### Strategy Comparison Tear Sheet")
    st.dataframe(comparison_df.style.format({
        "Annualized Return": "{:.2%}",
        "Annualized Volatility": "{:.2%}",
        "Sharpe Ratio": "{:.2f}",
        "Sortino Ratio": "{:.2f}",
        "Maximum Drawdown": "{:.2%}",
        "Beta": "{:.2f}",
        "Hist VaR (95%)": "{:.2%}",
        "Conditional VaR (95%)": "{:.2%}",
        "Diversification Ratio": "{:.2f}x"
    }), use_container_width=True)

    fig_comp = plot_portfolio_comparison(comparison_df)
    st.pyplot(fig_comp)
    plt.close(fig_comp)

    st.markdown("---")
    st.markdown("### Markowitz Efficient Frontier Optimization")
    with st.spinner("Optimizing mean-variance portfolios and running Monte Carlo simulation..."):
        try:
            frontier_res = generate_efficient_frontier(
                returns_df[valid_tickers], risk_free_rate=risk_free_rate, num_portfolios=2000
            )
            fig_frontier = plot_efficient_frontier(frontier_res, risk_free_rate=risk_free_rate)
            st.pyplot(fig_frontier)
            plt.close(fig_frontier)

            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                st.markdown("#### 🌟 Maximum Sharpe Portfolio")
                ms = frontier_res["max_sharpe"]
                st.write(f"**Annualized Return:** {ms['return']*100:.2f}% | **Volatility:** {ms['volatility']*100:.2f}% | **Sharpe:** {ms['sharpe']:.2f}")
                st.write("**Optimal Weights:**", {k: f"{v*100:.1f}%" for k, v in ms['weights'].items() if v > 0.01})

            with col_opt2:
                st.markdown("#### 🛡️ Minimum Volatility Portfolio")
                mv = frontier_res["min_volatility"]
                st.write(f"**Annualized Return:** {mv['return']*100:.2f}% | **Volatility:** {mv['volatility']*100:.2f}% | **Sharpe:** {mv['sharpe']:.2f}")
                st.write("**Optimal Weights:**", {k: f"{v*100:.1f}%" for k, v in mv['weights'].items() if v > 0.01})
        except Exception as e:
            st.warning(f"Could not compute efficient frontier: {e}")
