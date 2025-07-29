# train_model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from features import prepare_features  # or from features import prepare_features

df = prepare_features("AMZN", period="15y")
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df.dropna(inplace=True)

features = ["Prev Close", "5-day MA", "10-day MA", "Volume", "Return",
            "5-day Std", "Daily Range", "Open-Close", "High-Close",
            "RSI", "BB_High", "BB_Low", "BB_Width"]
X = df[features]
y = df["Target"]

train_size = int(len(X) * 0.8)
X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]

model = RandomForestClassifier()
model.fit(X_train, y_train)

acc = accuracy_score(y_val, model.predict(X_val))
print(f"Accuracy: {acc:.2f}")

joblib.dump(model, "model.pkl")
