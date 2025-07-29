📈 Stock Movement Predictor
This project is a machine learning web app that predicts whether a stock will go up or down tomorrow based on technical indicators. The user enters a ticker symbol (like AAPL), and the app uses real-time market data to make a prediction.

🚀 Features
- Predicts stock movement using a Random Forest Classifier
- Fetches live data from Yahoo Finance via yfinance
- Uses technical indicators like:
  - Moving averages (5-day, 10-day)
  - RSI (Relative Strength Index)
  - Bollinger Bands
  - Daily range, volume, return, and more
- Frontend built with HTML/CSS + JavaScript
- Backend built with Flask
- Displays model confidence, accuracy, and prediction

🧠 Model
- Binary classification: predicts if the next day’s close will be higher than today’s
- Trained on historical stock data (e.g. from 2010–present)
- Accuracy hovers around ~52–54% depending on ticker

🧰 Tech Stack
- Python
- scikit-learn
- pandas
- yfinance
- joblib
- ta (technical analysis)
- Flask
- HTML, CSS, JavaScript

📂 How to Run Locally

1. Clone the repo:
  git clone https://github.com/yourusername/stock-movement-predictor.git
  cd stock-movement-predictor

2. Install dependencies:
   pip install -r requirements.txt

3. Train the model:
   python train_model.py

4. Start the app:
  python app.py

5. Visit http://127.0.0.1:5000 in your browser
