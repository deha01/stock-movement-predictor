# movement_predictor.py (or features.py)

import yfinance as yf
import pandas as pd
import ta

def prepare_features(ticker, period="90d"):
    df = yf.download(ticker, period=period, auto_adjust=True)
    df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
    if df.empty:
        return pd.DataFrame()

    df["Prev Close"] = df["Close"].shift(1)
    df["5-day MA"] = df["Close"].rolling(window=5).mean().shift(1)
    df["10-day MA"] = df["Close"].rolling(window=10).mean().shift(1)
    df["Volume"] = df["Volume"].shift(1)
    df["Return"] = df["Close"].pct_change().shift(1)
    df["5-day Std"] = df["Close"].rolling(window=5).std().shift(1)
    df["Daily Range"] = (df["High"] - df["Low"]).shift(1)
    df["Open-Close"] = (df["Open"] - df["Close"]).shift(1)
    df["High-Close"] = (df["High"] - df["Close"]).shift(1)

    df['RSI'] = ta.momentum.RSIIndicator(close=df["Close"]).rsi()
    boll = ta.volatility.BollingerBands(close=df["Close"])
    df['BB_High'] = boll.bollinger_hband()
    df['BB_Low'] = boll.bollinger_lband()
    df['BB_Width'] = df['BB_High'] - df['BB_Low']

    df.dropna(inplace=True)
    return df
