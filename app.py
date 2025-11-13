from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Default tickers for the dashboard
DEFAULT_TICKERS = ['AAPL', 'MSFT', 'NFLX', 'GOOG']

# Advanced Prediction (Linear Regression)
def predict_next_close_ml(df):
    closes = df['Close'].values
    if len(closes) < 10:
        return round(float(closes[-1]), 2)
    X = np.arange(len(closes[-10:])).reshape(-1, 1)
    y = closes[-10:]
    model = LinearRegression()
    model.fit(X, y)
    next_day = np.array([[len(closes[-10:])]])
    predicted_price = model.predict(next_day)[0]
    return round(float(predicted_price), 2)

def plot_prices(selected_tickers, start_date, end_date):
    fig, ax = plt.subplots(figsize=(10, 6))
    for ticker in selected_tickers:
        df = yf.download(ticker, start=start_date, end=end_date)
        ax.plot(df.index, df["Close"], label=ticker)
    ax.set_title("Closing Prices of Stocks")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close(fig)
    return plot_url

def get_stock_summary_with_volatility_and_correlation(tickers, start_date, end_date):
    summary = []
    price_data = pd.DataFrame()

    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date)
        df['50MA'] = df['Close'].rolling(window=50).mean()
        df['200MA'] = df['Close'].rolling(window=200).mean()
        if df.empty:
            continue
        latest = df.iloc[-1]
        year_high = df['High'].max()
        year_low = df['Low'].min()
        ma50 = float(df['50MA'].dropna().iloc[-1]) if df['50MA'].dropna().size else 'NA'
        ma200 = float(df['200MA'].dropna().iloc[-1]) if df['200MA'].dropna().size else 'NA'
        predicted_close = predict_next_close_ml(df)

        daily_returns = df['Close'].pct_change().dropna()
        volatility = daily_returns.std() * (252 ** 0.5)

        summary.append({
            "Company": ticker,
            "Latest": round(float(latest['Close']), 2),
            "PredictedClose": predicted_close,
            "MA50": round(ma50, 2) if ma50 != 'NA' else 'NA',
            "MA200": round(ma200, 2) if ma200 != 'NA' else 'NA',
            "YearHigh": round(float(year_high), 2),
            "YearLow": round(float(year_low), 2),
            "Volatility": round(float(volatility), 4)
        })
        price_data[ticker] = df['Close']

    correlation_matrix = price_data.corr().round(4).to_dict()
    plot_url = plot_prices(tickers, start_date, end_date)
    return summary, correlation_matrix, plot_url

@app.route('/', methods=['GET'])
def index():
    selected_tickers = request.args.getlist('ticker') or DEFAULT_TICKERS
    start_date = request.args.get('start', '2025-08-01')
    end_date = request.args.get('end', '2025-11-06')

    summary, correlation_matrix, plot_url = get_stock_summary_with_volatility_and_correlation(
        tickers=selected_tickers, start_date=start_date, end_date=end_date)

    return render_template('index.html',
        summary=summary,
        correlation_matrix=correlation_matrix,
        tickers=DEFAULT_TICKERS,
        plot_url=plot_url,
        selected_tickers=selected_tickers,
        start_date=start_date,
        end_date=end_date
    )

if __name__ == "__main__":
    app.run(debug=True)