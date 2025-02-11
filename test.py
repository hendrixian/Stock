import requests
import pandas as pd

api_key = 'cuaibppr01qof06ijq70cuaibppr01qof06ijq7g'  # Replace with your valid API key
ticker = 'AAPL'
url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={api_key}"

try:
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200 and 'c' in data:
        # Create a DataFrame with the quote data
        df = pd.DataFrame({
            'Metric': ['Current Price', 'High Price', 'Low Price', 'Open Price', 'Previous Close'],
            'Value': [data['c'], data['h'], data['l'], data['o'], data['pc']]
        })
        print(df)
    else:
        print("No data found or an error occurred in the API response.")
except Exception as e:
    print(f"An error occurred: {e}")