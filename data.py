import yfinance as yf
import pandas as pd
import os

DIR = "data"

start_date = "2014-01-01"  
end_date = "2024-01-01"  
companies = {"google": "GOOGL", "dell": "DELL", "apple": "AAPL"}

for company, ticker in companies.items():
    data = yf.download(ticker, start= start_date, end= end_date)
    selected_columns = data[['High', 'Low', 'Open', 'Close']]
    selected_columns = selected_columns.dropna()

    metrics = {
    'Mean': selected_columns.mean(),
    'Highest': selected_columns.max(),
    'Lowest': selected_columns.min(),
    'Standard Deviation': selected_columns.std(),
    }

    metrics_df = pd.DataFrame(metrics)

    if not os.path.exists(DIR):
        os.makedirs(DIR, exist_ok=True) 

    metrics_df.to_csv(f"{DIR}/{company}_metrics.csv", index = False)

    selected_columns.to_csv(f"{DIR}/{company}.csv", index=False)