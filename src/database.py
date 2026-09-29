"""
Database Management Module.
Handles SQLite storage, indexing, and querying for stock metadata,
historical OHLCV data, portfolio allocations, and analytics results.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd


class DatabaseManager:
    """Manages SQLite database connections and schema operations."""

    def __init__(self, db_path: Union[str, Path] = "data/market_data.db"):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Returns a sqlite3 connection with row_factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Initializes tables and indexes if they do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Stocks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stocks (
                    ticker TEXT PRIMARY KEY,
                    company_name TEXT,
                    sector TEXT,
                    industry TEXT,
                    currency TEXT DEFAULT 'USD',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 2. Historical prices table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historical_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adjusted_close REAL NOT NULL,
                    volume INTEGER,
                    UNIQUE(ticker, date),
                    FOREIGN KEY (ticker) REFERENCES stocks (ticker)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_hist_ticker_date ON historical_prices(ticker, date)")

            # 3. Portfolio allocations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_allocations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_name TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    weight REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(portfolio_name, ticker)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_port_name ON portfolio_allocations(portfolio_name)")

            # 4. Analytics results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analytics_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_type TEXT NOT NULL,
                    target_name TEXT NOT NULL,
                    date_calculated TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analytics_target ON analytics_results(metric_type, target_name)")
            conn.commit()

    def save_stocks(self, stocks_data: List[Dict[str, str]]) -> None:
        """Upsert stock metadata records."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for s in stocks_data:
                cursor.execute("""
                    INSERT INTO stocks (ticker, company_name, sector, industry, currency)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(ticker) DO UPDATE SET
                        company_name = COALESCE(excluded.company_name, stocks.company_name),
                        sector = COALESCE(excluded.sector, stocks.sector),
                        industry = COALESCE(excluded.industry, stocks.industry),
                        currency = COALESCE(excluded.currency, stocks.currency)
                """, (
                    s.get("ticker", "").upper(),
                    s.get("company_name", s.get("ticker", "")),
                    s.get("sector", "Unknown"),
                    s.get("industry", "Unknown"),
                    s.get("currency", "USD")
                ))
            conn.commit()

    def save_historical_prices(self, df: pd.DataFrame) -> int:
        """
        Saves historical prices dataframe to SQLite.
        Expects columns: ['ticker', 'date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume']
        """
        if df.empty:
            return 0

        df_to_save = df.copy()
        # Standardize column names
        df_to_save.columns = [c.lower().replace(" ", "_") for c in df_to_save.columns]
        
        required_cols = {"ticker", "date", "adjusted_close"}
        if not required_cols.issubset(set(df_to_save.columns)):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df_to_save.columns:
                df_to_save[col] = None

        records = df_to_save[[
            "ticker", "date", "open", "high", "low", "close", "adjusted_close", "volume"
        ]].to_dict(orient="records")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT INTO historical_prices (ticker, date, open, high, low, close, adjusted_close, volume)
                VALUES (:ticker, :date, :open, :high, :low, :close, :adjusted_close, :volume)
                ON CONFLICT(ticker, date) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    adjusted_close = excluded.adjusted_close,
                    volume = excluded.volume
            """, records)
            conn.commit()
            return len(records)

    def get_historical_prices(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch long-format historical prices with optional filters.
        """
        query = "SELECT ticker, date, open, high, low, close, adjusted_close, volume FROM historical_prices WHERE 1=1"
        params: List[Union[str, int]] = []

        if tickers:
            placeholders = ",".join(["?"] * len(tickers))
            query += f" AND ticker IN ({placeholders})"
            params.extend([t.upper() for t in tickers])

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date ASC, ticker ASC"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
            return df

    def get_price_matrix(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        price_col: str = "adjusted_close"
    ) -> pd.DataFrame:
        """
        Returns a pivoted DataFrame with DateTime index and tickers as columns.
        """
        df = self.get_historical_prices(tickers=tickers, start_date=start_date, end_date=end_date)
        if df.empty:
            return pd.DataFrame()

        pivot_df = df.pivot(index="date", columns="ticker", values=price_col)
        pivot_df.sort_index(inplace=True)
        return pivot_df

    def save_portfolio_allocation(self, portfolio_name: str, weights: Dict[str, float]) -> None:
        """Saves a named portfolio allocation."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM portfolio_allocations WHERE portfolio_name = ?", (portfolio_name,))
            for ticker, weight in weights.items():
                cursor.execute("""
                    INSERT INTO portfolio_allocations (portfolio_name, ticker, weight)
                    VALUES (?, ?, ?)
                """, (portfolio_name, ticker.upper(), float(weight)))
            conn.commit()

    def get_portfolio_allocations(self, portfolio_name: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Retrieve portfolio allocations by name, or all saved portfolios."""
        query = "SELECT portfolio_name, ticker, weight FROM portfolio_allocations"
        params: List[str] = []
        if portfolio_name:
            query += " WHERE portfolio_name = ?"
            params.append(portfolio_name)
        query += " ORDER BY portfolio_name, ticker"

        portfolios: Dict[str, Dict[str, float]] = {}
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            for row in cursor.fetchall():
                p_name = row["portfolio_name"]
                if p_name not in portfolios:
                    portfolios[p_name] = {}
                portfolios[p_name][row["ticker"]] = float(row["weight"])
        return portfolios

    def save_analytics_result(
        self, metric_type: str, target_name: str, date_calculated: str, metrics: Dict
    ) -> None:
        """Save computed analytics JSON payload."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analytics_results (metric_type, target_name, date_calculated, metrics_json)
                VALUES (?, ?, ?, ?)
            """, (metric_type, target_name, date_calculated, json.dumps(metrics)))
            conn.commit()

    def get_available_tickers(self) -> List[str]:
        """Returns list of distinct tickers available in the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT ticker FROM historical_prices ORDER BY ticker")
            return [row[0] for row in cursor.fetchall()]
