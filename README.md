📈 Predictive Stock Insights Dashboard

### 👩‍💻 Developed by [Chandana M](https://github.com/chandanamshetty330-coder)

A Flask-based web dashboard that provides real-time and predictive insights on stock market data using machine learning, data analysis, and visualization.
This project helps users analyze multiple stock trends, predict future prices, and visualize historical patterns with interactive charts.


---

🚀 Features

📊 Stock Summary Table — Displays latest price, 50-day and 200-day moving averages, yearly high/low, and volatility.

🤖 Machine Learning Prediction — Uses Linear Regression to forecast the next closing price.

🔗 Multiple Stock Comparison — Compare selected companies’ performance over custom date ranges.

📉 Volatility and Correlation Matrix — Understand how stocks move in relation to one another.

🖼 Dynamic Graphs — Auto-generated visual comparison charts using Matplotlib.

🌐 Responsive Interface — Clean and professional dashboard layout for both desktop and mobile.



---

🧠 Technologies Used

Category	Tools

Backend Framework	Flask
Data Source	Yahoo Finance (yfinance)
Data Analysis	Pandas, NumPy
Machine Learning	Scikit-learn (Linear Regression)
Visualization	Matplotlib
Frontend	HTML, CSS (Responsive Design)

⚙ Installation & Setup

1. Clone the repository

git clone https://github.com/chandanamshetty330-coder/stockmarket-data-analysis.git
cd stock-market-data-analysis


2. Install dependencies

pip install flask yfinance pandas matplotlib scikit-learn


3. Run the Flask app

python app.py


4. Open in your browser:

http://127.0.0.1:5000/


📂 Project Structure

📁 stock-market-data-analysis
├── app.py                # Flask backend logic
├── templates/
│   └── index.html        # Frontend HTML dashboard
├── static/
│   └── style.css         # Styling (optional)
└── README.md             # Project documentation


---

📊 Output Preview

Stock Summary Table — Real-time market data

Correlation Matrix — Relationship between selected tickers

Price Comparison Chart — Line graph visualization



---

🧩 Example Tickers

AAPL, MSFT, GOOG, NFLX


---

💡 Future Enhancements

Add live stock news updates

Include portfolio profit/loss tracking

Support for crypto and global indices

Use advanced ML models (LSTM, ARIMA) for better prediction accuracy



---

👩‍💻 Developer Information

Developer: Chandana M
📧 Email: chandanamshetty330@gmail.com
🔗 LinkedIn: linkedin.com/in/chandana-m-b5966b368
💻 GitHub: github.com/chandanamshetty330-coder
