import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
import ta


df = yf.download("AAPL", start="2010-01-01", auto_adjust=True)
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# Old features
df["Prev Close"] = df["Close"].shift(1)
df["5-day MA"] = df["Close"].rolling(window=5).mean().shift(1)
df["10-day MA"] = df["Close"].rolling(window=10).mean().shift(1)
df["Volume"] = df["Volume"].shift(1)

# New features
df["Return"] = df["Close"].pct_change().shift(1)
df["5-day Std"] = df["Close"].rolling(window=5).std().shift(1)
df["Daily Range"] = (df["High"] - df["Low"]).shift(1)
df["Open-Close"] = (df["Open"] - df["Close"]).shift(1)
df["High-Close"] = (df["High"] - df["Close"]).shift(1)


# RSI Features
df.columns = df.columns.get_level_values(0)  # Flattens columns
df['RSI'] = ta.momentum.RSIIndicator(close=df["Close"]).rsi()

# Bollinger Bands
bollinger = ta.volatility.BollingerBands(close=df["Close"])
df['BB_High'] = bollinger.bollinger_hband()
df['BB_Low'] = bollinger.bollinger_lband()
df['BB_Width'] = df['BB_High'] - df['BB_Low']

# MACD Features
df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = df["EMA_12"] - df["EMA_26"]

df.dropna(inplace = True)

features = ["Prev Close", "5-day MA", "10-day MA", "Volume", "Return", "5-day Std", "Daily Range", "Open-Close", "High-Close", "RSI", "BB_High", "BB_Low", "BB_Width"]
targets = ["Target"]

X = df[features]
y = df["Target"]

# Split the data
#X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 0)

train_size = int(len(X) * 0.8)
X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]

# Model
#model = RandomForestClassifier(n_estimators=300, random_state=0)
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_val)

#mae = mean_absolute_error(y_val, preds)
acc = accuracy_score(y_val, preds)

print(f"Accuracy: {acc:.2f}")

# dummy = DummyClassifier(strategy="most_frequent")
# dummy.fit(X_train, y_train)
# baseline_preds = dummy.predict(X_val)
# print("Baseline accuracy:", accuracy_score(y_val, baseline_preds))