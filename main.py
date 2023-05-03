import requests
import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

import datetime

csv_filename = 'bitcoin_prices.csv'  # define the path and filename of the CSV file to write to

# Define start and end dates for API request and CSV file
api_start_date = '2018-01-27'
api_end_date = datetime.date.today().strftime('%Y-%m-%d')

# Make API request for bitcoin prices
try:
    url = f'https://api.coindesk.com/v1/bpi/historical/close.json?start={api_start_date}&end={api_end_date}'
    response = requests.get(url)
    response.raise_for_status()   # raise an exception if the response is not successful
    data = response.json()
except requests.exceptions.RequestException as e:
    print('Error while making API request:', e)
    exit()

# Extract date and price data from the response and store them in a list
price_data = [['Date', 'Price']]
try:
    for date, price in data['bpi'].items():
        price_data.append([date, price])
except (KeyError, TypeError) as e:
    print('Error while extracting data from API response:', e)
    exit()

# If there are more pages, keep making requests and adding data to the list
while 'next_page' in data:
    try:
        url = data['next_page']
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        for date, price in data['bpi'].items():
            price_data.append([date, price])
    except (requests.exceptions.RequestException, KeyError, TypeError) as e:
        print('Error while processing API response:', e)
        exit()

# Write the data to a CSV file
try:
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(price_data)
except IOError as e:
    print('Error while writing data to CSV file:', e)
    exit()

# Load data from CSV file into a dataframe
try:
    with open(csv_filename, mode='r') as csv_file:
        bitcoin_df = pd.read_csv(csv_file)
except (IOError, pd.errors.EmptyDataError) as e:
    print('Error while loading data from CSV file:', e)
    exit()

# Convert 'Price' column to numeric values
try:
    bitcoin_df['Price'] = pd.to_numeric(bitcoin_df['Price'], errors='coerce')
except ValueError as e:
    print('Error while converting data to numeric values:', e)
    exit()

# Convert 'Date' column to datetime objects
try:
    bitcoin_df['Date'] = pd.to_datetime(bitcoin_df['Date'])
except ValueError as e:
    print('Error while converting data to datetime objects:', e)
    exit()

# Drop rows with missing or invalid data
bitcoin_df.dropna(inplace=True)

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(bitcoin_df[['Price']])

# Convert the scaled data to a dataframe
bitcoin_df_scaled = pd.DataFrame(scaled_data, columns=['Price'], index=bitcoin_df.index)

# Split data into training and testing sets
train_data = bitcoin_df_scaled[:int(0.8 * len(bitcoin_df))]
test_data = bitcoin_df_scaled[int(0.8 * len(bitcoin_df)):]

# Fit linear regression model
regressor = LinearRegression()
X_train = train_data.index.values.reshape(-1, 1)
y_train = train_data['Price'].values.reshape(-1, 1)
regressor.fit(X_train, y_train)

# Make predictions for next 20 days
X_pred = np.arange(len(bitcoin_df), len(bitcoin_df) + 20).reshape(-1, 1)
y_pred_scaled = regressor.predict(X_pred)
y_pred = scaler.inverse_transform(y_pred_scaled)

# Convert the date range for the next 20 days to datetime objects
dates = pd.date_range(start=bitcoin_df['Date'].max(), periods=20, freq='D')

# Plot the data and predictions
plt.figure(figsize=(12, 6))
plt.plot(bitcoin_df['Date'], bitcoin_df['Price'], label='Actual')
plt.plot(dates, y_pred, label='Predicted')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# Print predicted price for the next 20 days
for pred in y_pred:
    print(f"{pred[0]:.2f}")

# Plot the Bitcoin price data
sns.set_style("darkgrid")
plt.figure(figsize=(12, 6))
plt.plot(bitcoin_df.index, bitcoin_df['Price'], label='Actual')
plt.title('Bitcoin Price Data', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.legend(fontsize=12)

# Plot the predicted prices for the next 20 days
plt.figure(figsize=(12, 6))
plt.plot(X_pred, y_pred, label='Predicted')
plt.title('Bitcoin Price Predictions', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.legend(fontsize=12)

# Show the plots
plt.show()

