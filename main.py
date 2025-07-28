import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

df = yf.download("AAPL", period="2y", auto_adjust=True)
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

df.dropna(inplace = True)

X = df[["Prev Close", "5-day MA", "10-day MA", "Volume", "Return", "5-day Std", "Daily Range", "Open-Close", "High-Close"]]
y = df["Target"]

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 0)

# Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_val)

#mae = mean_absolute_error(y_val, preds)
acc = accuracy_score(y_val, preds)

print(f"Accuracy: {acc:.2f}")