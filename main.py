import requests
import streamlit as st
import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Page layout
st.set_page_config(layout="wide")

# Title
st.title('Bitcoin Price Prediction')

# Introduction and overview
st.markdown("""
This app retrieves the list of the **Bitcoin** prices for the last 3 years from the [Coindesk](https://www.coindesk.com/price/bitcoin) website and predicts the price for the next 20 days using a **Linear Regression** model.
""")
st.markdown("""
* **Python libraries:** pandas, streamlit, numpy, sklearn, matplotlib, seaborn
* **Data source:** [Coindesk](https://www.coindesk.com/price/bitcoin)
""")
st.markdown("---")

# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Collect user input features'):
    selected_year = st.sidebar.selectbox('Year', list(reversed(range(2018, 2021))))

# Get Bitcoin prices
url = 'https://api.coindesk.com/v1/bpi/historical/close.json'
params = {'start': f'{selected_year}-01-01', 'end': f'{selected_year}-12-31'}
response = requests.get(url, params=params)
data = response.json()['bpi']
df = pd.DataFrame.from_dict(data, orient='index', columns=['price'])

# Convert prices to the desired format
df.index = pd.to_datetime(df.index)
df.index.name = 'date'
df['price'] = df['price'].astype('float')

# Keep only the next 20 days
df_future = df.tail(20)

# Display Bitcoin prices chart
st.subheader(f'2. Bitcoin Prices chart for the year {selected_year}')
st.line_chart(df)

# Display prices for the next 20 days
st.subheader('3. Bitcoin Prices for the next 20 days')
st.table(df_future)

csv_filename = 'bitcoin_prices.csv'  # name of the CSV file to write to

# Define start and end dates for API request and CSV file
api_start_date = '2018-01-27'
api_end_date = datetime.date.today().strftime('%Y-%m-%d')

# Make API request for bitcoin prices
# Get the data from the API
try:
    # construct the url
    url = f'https://api.coindesk.com/v1/bpi/historical/close.json?start={api_start_date}&end={api_end_date}'

    # send the request
    response = requests.get(url)
    response.raise_for_status()   # raise an exception if the response is not successful

    # convert the response to json
    data = response.json()

# catch any errors that may occur
except requests.exceptions.RequestException as e:
    print('Error while making API request:', e)
    exit()

# Extract date and price data from the response and store them in a list
# Create a list of lists with date and price data
price_data = [['Date', 'Price']]

# Extract date and price data from response and append to price_data
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
    # Open a file to write the CSV data to
    with open(csv_filename, mode='w', newline='') as csv_file:
        # Create a CSV writer object
        writer = csv.writer(csv_file)
        # Write the data to the CSV file
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
bitcoin_df['Price'] = pd.to_numeric(bitcoin_df['Price'], errors='coerce')
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


