import yfinance as yf
import pandas as pd
import numpy as np


def fetch_stock_data(ticker="AAPL", period="2y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)

    # Simple Moving Averages
    df["MA_7"]  = df["Close"].rolling(window=7).mean()
    df["MA_21"] = df["Close"].rolling(window=21).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()

    # Exponential Moving Averages
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]

    # RSI
    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs       = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    rolling_std   = df["Close"].rolling(20).std()
    df["BB_upper"] = df["MA_21"] + (2 * rolling_std)
    df["BB_lower"] = df["MA_21"] - (2 * rolling_std)
    df["BB_width"] = df["BB_upper"] - df["BB_lower"]

    # Price & Volume changes
    df["Price_Change"]  = df["Close"].pct_change() * 100
    df["Volume_Change"] = df["Volume"].pct_change() * 100
    df["Log_Volume"]    = np.log1p(df["Volume"])

    df.dropna(inplace=True)
    return df


def get_company_info(ticker="AAPL"):
    try:
        info = yf.Ticker(ticker).info
        return {
            "name":       info.get("longName", ticker),
            "sector":     info.get("sector", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "currency":   info.get("currency", "USD"),
        }
    except Exception:
        return {"name": ticker, "sector": "N/A", "market_cap": 0, "currency": "USD"}