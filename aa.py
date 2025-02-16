import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
from datetime import datetime

# Fetch stock data with month/day/year format
print("hello world")
def get_stock_data(ticker, start, end):
    # Convert string dates from month/day/year format to datetime objects
    start_date = datetime.strptime(start, '%m/%d/%Y')  # Using / instead of .
    end_date = datetime.strptime(end, '%m/%d/%Y')  # Using / instead of .
    
    stock = yf.download(ticker, start=start_date, end=end_date)
    return stock[['Close']]


# Prepare Data for LSTM Model
def prepare_data(stock_data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data[['Close']])
    
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    return X.reshape(X.shape[0], X.shape[1], 1), y, scaler

# Create LSTM Model
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Predict Future Stock Prices
def predict_stock_price(model, scaler, stock_data, time_step=60):
    test_input = scaler.transform(stock_data[['Close']].values)
    X_test = []
    for i in range(time_step, len(test_input)):
        X_test.append(test_input[i-time_step:i, 0])
    X_test = np.array(X_test).reshape(len(X_test), time_step, 1)
    predictions = model.predict(X_test)
    return scaler.inverse_transform(predictions)

# UI Functionality
def run_prediction():
    stock_symbol = stock_entry.get().upper()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    
    if not stock_symbol:
        messagebox.showerror("Error", "Please enter a stock symbol.")
        return
    
    try:
        stock_data = get_stock_data(stock_symbol, start_date, end_date)
        X, y, scaler = prepare_data(stock_data)
        
        model = create_lstm_model((X.shape[1], 1))
        model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
        
        predictions = predict_stock_price(model, scaler, stock_data)
        
        plt.figure(figsize=(12, 6))
        plt.plot(stock_data.index[-len(predictions):], stock_data['Close'].values[-len(predictions):], label='Actual Price', color='blue')
        plt.plot(stock_data.index[-len(predictions):], predictions, label='Predicted Price', color='red')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title(f'{stock_symbol} Stock Price Prediction')
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create UI
root = tk.Tk()
root.title("Stock Price Prediction")
root.geometry("400x300")

tk.Label(root, text="Stock Symbol:").pack(pady=5)
stock_entry = tk.Entry(root)
stock_entry.pack(pady=5)

tk.Label(root, text="Start Date:").pack(pady=5)
# Set the date format to show full year (yyyy)
start_date_entry = DateEntry(root, width=12, background='darkblue', foreground='white', borderwidth=2, date_pattern='mm/dd/yyyy')
start_date_entry.pack(pady=5)

tk.Label(root, text="End Date:").pack(pady=5)
# Set the date format to show full year (yyyy)
end_date_entry = DateEntry(root, width=12, background='darkblue', foreground='white', borderwidth=2, date_pattern='mm/dd/yyyy')
end_date_entry.pack(pady=5)

tk.Button(root, text="Predict", command=run_prediction).pack(pady=20)

root.mainloop()
print("mkm")
print("paw")
print ("may")