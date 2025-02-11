import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# Fetch stock data
def get_stock_data(ticker, start, end):
    stock = yf.download(ticker, start=start, end=end)
    return stock[['Close']]

# Prepare data for ML models
def prepare_data(df, lookback=30):
    X, y = [], []
    for i in range(len(df) - lookback):
        X.append(df[i:i+lookback])
        y.append(df[i+lookback])
    return np.array(X), np.array(y)

# Load data
ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2024-01-01'
data = get_stock_data(ticker, start_date, end_date)

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Prepare train-test split
lookback = 30
X, y = prepare_data(data_scaled, lookback)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
y_pred_rf = rf_model.predict(X_test.reshape(X_test.shape[0], -1))

# LSTM Model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Predictions
y_pred_lstm = lstm_model.predict(X_test)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot results
plt.figure(figsize=(10,5))
plt.plot(y_test_actual, label='Actual Price')
plt.plot(y_pred_lstm, label='LSTM Prediction')
plt.legend()
plt.title('Stock Price Prediction')
plt.show()
