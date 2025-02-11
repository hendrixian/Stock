from flask import Flask, render_template, request
from flask_caching import Cache
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os
from datetime import datetime, timedelta
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from matplotlib.dates import DateFormatter, YearLocator

app = Flask(__name__)

app.config ['CACHE_TYPE'] = 'simple'
app.config['CACHE_DEFAULT_TIMEOUT'] = 86400  # Cache timeout (24 hours)
cache = Cache(app)
API_KEY = 'LWY10CXI18SDUVBQ'  # Replace with your valid Alpha Vantage API key

@cache.cached(timeout=86400, key_prefix='stock_data_{ticker}')
def fetch_stock_data(ticker):
    """Fetch historical stock data for a given ticker using Alpha Vantage API."""
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={API_KEY}&outputsize=full"
    response = requests.get(url)

    # Log the raw response for debugging
    if response.status_code != 200:
        raise ValueError(f"Error: API request failed with status code {response.status_code}")

    data = response.json()

    if 'Error Message' in data:
        raise ValueError(f"API Error: {data['Error Message']}")

    if 'Note' in data and 'Invalid API call' in data['Note']:
        raise ValueError("API Error: Invalid API call. Please retry or visit the documentation for TIME_SERIES_DAILY.")

    if 'Time Series (Daily)' in data:
        time_series = data['Time Series (Daily)']
        dates = []
        close_prices = []
        high_prices = []
        low_prices = []
        open_prices = []

        for date, values in time_series.items():
            dates.append(date)
            close_prices.append(float(values['4. close']))
            high_prices.append(float(values['2. high']))
            low_prices.append(float(values['3. low']))
            open_prices.append(float(values['1. open']))

        return {
            't': dates,
            'c': close_prices,
            'h': high_prices,
            'l': low_prices,
            'o': open_prices,
        }
    else:
        raise ValueError("Error: No data found for the ticker symbol.")
    
def predict_future_with_svd(df, future_days=30):
    """
    Use SVD to analyze stock data and predict future prices.
    """
    # Extract the features to be analyzed
    prices_matrix = df[['Close', 'High', 'Low', 'Open']].values

    # Apply SVD
    n_components = min(2, len(prices_matrix))  # Ensure n_components doesn't exceed rows
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(prices_matrix)
    singular_values = svd.singular_values_
    components = svd.transform(prices_matrix)

    # Linear regression for each component to predict future trends
    future_dates = pd.date_range(start=df.index[-1], periods=future_days + 1, freq='D')[1:]
    future_dates_ordinal = future_dates.map(datetime.toordinal)

    predictions = []
    for i in range(components.shape[1]):  # Iterate over components
        X = df.index.map(datetime.toordinal).values.reshape(-1, 1)
        y = components[:, i]
        model = LinearRegression()
        model.fit(X, y)

        # Predict future components
        future_components = model.predict(future_dates_ordinal.values.reshape(-1, 1))
        predictions.append(future_components)
        print("testing....")
        print("test 2.....")
    # Reconstruct future prices from predicted components
    predicted_matrix = svd.inverse_transform(np.column_stack(predictions))
    future_df = pd.DataFrame(
        predicted_matrix,
        columns=['Close', 'High', 'Low', 'Open'],
        index=future_dates
    )

    return future_df, singular_values


def analyze_stock(ticker):
    try:
        stock_data = fetch_stock_data(ticker)

        # Create DataFrame and sort by Date
        df = pd.DataFrame({
            'Date': pd.to_datetime(stock_data['t'], format='%Y-%m-%d'),
            'Close': stock_data['c'],
            'High': stock_data['h'],
            'Low': stock_data['l'],
            'Open': stock_data['o'],
        }).sort_values(by='Date')

        df.set_index('Date', inplace=True)

        # Drop rows with missing values to avoid SVD errors
        df.dropna(inplace=True)

        # Perform SVD-based future predictions
        future_days = 7
        future_df, singular_values = predict_future_with_svd(df, future_days=future_days)

        # Merge current and future data for plotting
        combined_df = pd.concat([df, future_df])

        # Prices matrix for SVD
        prices_matrix = df[['Close', 'High', 'Low', 'Open']].values

        # Ensure SVD components do not exceed the number of rows
        n_components = min(2, len(prices_matrix))
        if n_components < 1:
            raise ValueError("Not enough data points for SVD analysis.")

        # Apply SVD
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(prices_matrix)

        singular_values = svd.singular_values_

        approximated_prices = svd.inverse_transform(svd.transform(prices_matrix))
        df['Predicted_Close'] = approximated_prices[:, 0]

        # Check if singular values are empty
        if len(singular_values) == 0:
            raise ValueError("SVD did not return any singular values.")

        # Linear Regression
        df['Date_ordinal'] = df.index.map(datetime.toordinal)
        X = df['Date_ordinal'].values.reshape(-1, 1)
        y = df['Close'].values
        model = LinearRegression()
        model.fit(X, y)
        df['Trend'] = model.predict(X)

        # Holt-Winters Forecasting
        model_hw = ExponentialSmoothing(df['Close'], trend='add', seasonal=None, damped_trend=True)
        model_hw_fit = model_hw.fit()
        df['Forecast'] = model_hw_fit.fittedvalues

        # Residuals (errors)
        df['Residuals'] = df['Close'] - df['Predicted_Close']

        # Create subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 24), sharex=True)  # Changed to 4 subplots

        # Set x-axis limits to start from the first date in the DataFrame
        start_date = df.index.min()
        end_date = df.index.max()
        for ax in axes:
            ax.set_xlim(start_date, end_date)  # Set x-axis limits

        # Format x-ticks to show every year
        for ax in axes:
            ax.xaxis.set_major_locator(YearLocator())  # Set major ticks to every year
            ax.xaxis.set_major_formatter(DateFormatter('%Y'))  # Show years on x-axis

        # 1. Actual vs Predicted
        axes[0].plot(df.index, df['Close'], label='Actual Close Price', color='blue')
        axes[0].plot(df.index, df['Predicted_Close'], label='SVD Predicted Close Price', color='orange')
        axes[0].set_title(f'{ticker} - Actual vs Predicted Close Price')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Stock Price (Currency)')
        axes[0].legend()
        axes[0].grid(True)
        plt.savefig(f'static/{ticker}_actual_vs_predicted.png')

        # 2. Trend and Forecast
        axes[1].plot(df.index, df['Close'], label='Actual Close Price', color='blue')
        axes[1].plot(df.index, df['Trend'], label='Linear Regression Trend', color='green')
        axes[1].plot(df.index, df['Forecast'], label='Holt-Winters Forecast', color='purple')
        axes[1].set_title(f'{ticker} - Trend and Forecast')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Stock Price (Currency)')
        axes[1].legend()
        axes[1].grid(True)
        plt.savefig(f'static/{ticker}_trend_forecast.png')

        # 3. Residuals
        axes[2].plot(df.index, df['Residuals'], label='Residuals (Actual - Predicted)', color='red')
        axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[2].set_title(f'{ticker} - Residuals (Errors)')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Residual Value')
        axes[2].legend()
        axes[2].grid(True)
        plt.savefig(f'static/{ticker}_residuals.png')

        # 4. Full Combined Data
        axes[3].plot(combined_df.index, combined_df['Close'], label='Combined Prices', color='purple')
        axes[3].set_title(f'{ticker} - Full Data (Including Future Predictions)')
        axes[3].set_xlabel('Date')
        axes[3].set_ylabel('Stock Price')
        axes[3].legend()
        axes[3].grid(True)
        plt.savefig(f'static/{ticker}_full_data.png')

        plt.close()

        return ticker

    except Exception as e:
        return str(e)

@app.route('/', methods=['GET', 'POST'])
def index():
    error_message = None
    ticker = None
    if request.method == 'POST':
        ticker = request.form['ticker']
        result = analyze_stock(ticker)
        if isinstance(result, str) and "Error" in result:
            error_message = result
            ticker = None
        else:
            ticker = result
    return render_template('index.html', ticker=ticker, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
