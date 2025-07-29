from flask import Flask, request, jsonify, render_template
import joblib
from features import prepare_features

app = Flask(__name__)
model, model_accuracy = joblib.load("model.pkl")

features = [
    "Prev Close", "5-day MA", "10-day MA", "Volume", "Return", "5-day Std",
    "Daily Range", "Open-Close", "High-Close", "RSI", "BB_High", "BB_Low", "BB_Width"
]

@app.route("/")
def home():
    return render_template("index.html", accuracy=round(model_accuracy * 100, 2))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        ticker = data.get("ticker", "").upper()

        if not ticker:
            return jsonify({"error": "Ticker is required."})

        df = prepare_features(ticker)

        if df.empty or not all(col in df.columns for col in features):
            return jsonify({"error": "Not enough data for prediction."})

        latest = df.iloc[-1:][features]
        proba = model.predict_proba(latest)[0]
        prediction = model.predict(latest)[0]
        confidence = round(max(proba) * 100, 2) # Confidence in percentage

        return jsonify({
            "prediction": int(prediction),
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
