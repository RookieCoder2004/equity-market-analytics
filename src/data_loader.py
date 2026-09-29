"""
Data Ingestion and Preprocessing Module.
Fetches historical market data via yfinance, cleans and standardizes data,
and caches it into the SQLite database. Includes an offline simulation fallback.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import yfinance as yf

from src.database import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "JPM", "META"]
DEFAULT_BENCHMARK = "^GSPC"


class MarketDataLoader:
    """Ingests, cleans, and stores equity market data."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()

    def fetch_and_store_ticker_data(
        self,
        tickers: List[str],
        start_date: Union[str, datetime] = "2020-01-01",
        end_date: Optional[Union[str, datetime]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetches historical data for multiple tickers, cleans it, and persists to DB.
        Returns long-form combined DataFrame.
        """
        if isinstance(start_date, datetime):
            start_date_str = start_date.strftime("%Y-%m-%d")
        else:
            start_date_str = str(start_date)

        if end_date is None:
            end_date_str = datetime.today().strftime("%Y-%m-%d")
        elif isinstance(end_date, datetime):
            end_date_str = end_date.strftime("%Y-%m-%d")
        else:
            end_date_str = str(end_date)

        tickers = [t.strip().upper() for t in tickers if t.strip()]
        if not tickers:
            raise ValueError("No valid tickers provided.")

        all_records = []
        stocks_metadata = []

        for ticker in tickers:
            df_ticker = None
            if use_cache:
                cached_df = self.db.get_historical_prices(
                    tickers=[ticker], start_date=start_date_str, end_date=end_date_str
                )
                if not cached_df.empty and len(cached_df) > 10:
                    logger.info(f"Loaded {len(cached_df)} cached records for {ticker}")
                    all_records.append(cached_df)
                    continue

            # Fetch via yfinance
            try:
                logger.info(f"Downloading {ticker} from {start_date_str} to {end_date_str} via yfinance...")
                yf_ticker = yf.Ticker(ticker)
                raw_df = yf_ticker.history(start=start_date_str, end=end_date_str, auto_adjust=False)

                if raw_df is not None and not raw_df.empty:
                    df_ticker = self._clean_raw_ticker_data(ticker, raw_df)
                    info = {}
                    try:
                        info = yf_ticker.info or {}
                    except Exception:
                        pass
                    stocks_metadata.append({
                        "ticker": ticker,
                        "company_name": info.get("shortName") or info.get("longName") or ticker,
                        "sector": info.get("sector") or "Equity",
                        "industry": info.get("industry") or "Financial Market",
                        "currency": info.get("currency") or "USD"
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker} from yfinance: {e}")

            # Fallback generator if empty or failed
            if df_ticker is None or df_ticker.empty:
                logger.warning(f"Generating realistic synthetic fallback for {ticker}...")
                df_ticker = self._generate_synthetic_ticker_data(ticker, start_date_str, end_date_str)
                stocks_metadata.append({
                    "ticker": ticker,
                    "company_name": f"{ticker} Corporation",
                    "sector": "Technology" if ticker in ["AAPL", "MSFT", "GOOGL", "NVDA", "META"] else "Diversified",
                    "industry": "Consumer Electronics" if ticker == "AAPL" else "Equities",
                    "currency": "USD"
                })

            if df_ticker is not None and not df_ticker.empty:
                self.db.save_historical_prices(df_ticker)
                all_records.append(df_ticker)

        if stocks_metadata:
            self.db.save_stocks(stocks_metadata)

        if not all_records:
            return pd.DataFrame()

        combined_df = pd.concat(all_records, ignore_index=True)
        combined_df.drop_duplicates(subset=["ticker", "date"], inplace=True)
        combined_df.sort_values(by=["date", "ticker"], inplace=True)
        return combined_df

    def _clean_raw_ticker_data(self, ticker: str, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes raw yfinance DataFrame into clean SQLite structure."""
        df = raw_df.copy()
        # Reset index to extract Date column
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
            df["date"] = df.index.strftime("%Y-%m-%d")
        elif "Date" in df.columns:
            df["date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        else:
            df["date"] = pd.to_datetime(df.index).strftime("%Y-%m-%d")

        # Map column names
        col_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjusted_close",
            "Volume": "volume"
        }
        df.rename(columns=col_map, inplace=True)

        if "adjusted_close" not in df.columns and "close" in df.columns:
            df["adjusted_close"] = df["close"]

        df["ticker"] = ticker.upper()

        # Handle missing values
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.dropna(subset=["adjusted_close"], inplace=True)

        # Ensure correct column types
        for col in ["open", "high", "low", "close", "adjusted_close"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0).astype(int)

        expected_cols = ["ticker", "date", "open", "high", "low", "close", "adjusted_close", "volume"]
        available_cols = [c for c in expected_cols if c in df.columns]
        return df[available_cols].copy()

    def _generate_synthetic_ticker_data(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Generates realistic geometric Brownian motion price series for fallback/testing.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq="B")  # Business days
        if len(dates) == 0:
            dates = pd.date_range(end=datetime.today(), periods=252, freq="B")

        np.random.seed(abs(hash(ticker)) % (2**32))
        n = len(dates)

        # Realistic parameter ranges
        base_prices = {
            "AAPL": 150.0, "MSFT": 280.0, "GOOGL": 120.0, "AMZN": 130.0,
            "NVDA": 200.0, "TSLA": 180.0, "JPM": 140.0, "META": 250.0,
            "^GSPC": 4000.0, "SPY": 400.0
        }
        s0 = base_prices.get(ticker, 100.0)
        mu = 0.12  # Expected annual return 12%
        sigma = 0.22 if ticker != "^GSPC" else 0.15  # Annual vol 22% (15% for index)

        dt = 1.0 / 252.0
        daily_drift = (mu - 0.5 * sigma**2) * dt
        daily_vol = sigma * np.sqrt(dt)

        shock = np.random.normal(0, 1, n)
        log_ret = daily_drift + daily_vol * shock
        price_path = s0 * np.exp(np.cumsum(log_ret))

        high = price_path * (1 + np.abs(np.random.normal(0.005, 0.003, n)))
        low = price_path * (1 - np.abs(np.random.normal(0.005, 0.003, n)))
        open_p = low + (high - low) * np.random.uniform(0.2, 0.8, n)
        volume = np.random.randint(5_000_000, 50_000_000, n)

        df = pd.DataFrame({
            "ticker": ticker.upper(),
            "date": dates.strftime("%Y-%m-%d"),
            "open": open_p.round(2),
            "high": high.round(2),
            "low": low.round(2),
            "close": price_path.round(2),
            "adjusted_close": price_path.round(2),
            "volume": volume
        })
        return df

    def get_adjusted_close_matrix(
        self,
        tickers: List[str],
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Helper returning a clean price matrix DataFrame (Date index x Tickers).
        """
        # Ensure data is loaded/updated
        self.fetch_and_store_ticker_data(tickers, start_date=start_date, end_date=end_date)
        matrix = self.db.get_price_matrix(
            tickers=tickers, start_date=start_date, end_date=end_date, price_col="adjusted_close"
        )
        return matrix.ffill().bfill().dropna()
