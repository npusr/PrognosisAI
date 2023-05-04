# Final year project - PrognosisAI - Bitcoin Price Predictor
# Date created: 06/03/2023
# Description: This is the main file for the PrognosisAI - Bitcoin Price Predictor web app.
# Coder: Antonis Platis
# Version: v1.0.0

# Import libraries
import requests  # to make HTTP requests
import streamlit as st  # to build the web app
import csv  # to read and write CSV files
import pandas as pd  # to process dataframes
import numpy as np  # to process arrays
from sklearn.linear_model import LinearRegression  # to build the model
from sklearn.preprocessing import MinMaxScaler  # to scale the data
import matplotlib.pyplot as plt  # to plot charts
import seaborn as sns  # to plot charts
import datetime  # to process dates

# Set page config

# Page layout
st.set_page_config(layout="wide")  # wide mode

# Title
st.title('PrognosisAI - Bitcoin Price Predictor')  # title of the app

# Introduction and overview
st.markdown("""
This app retrieves the list of the **Bitcoin** prices for the last 3 years (2018-2020) from the [Coindesk](https://www.coindesk.com/price/bitcoin) website and predicts the price for the next 20 days using a **Linear Regression** model.
""")  # description of the app
st.markdown("""
* **Python libraries:** pandas, streamlit, numpy, sklearn, matplotlib, seaborn
* **Data source:** [Coindesk](https://www.coindesk.com/price/bitcoin)
""")  # description of the app
st.markdown("---")  # separator

# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Select the year'):  # sidebar title
    selected_year = st.sidebar.selectbox('Year', list(reversed(range(2018, 2021))))  # selectbox for the year

# Get Bitcoin prices
url = 'https://api.coindesk.com/v1/bpi/historical/close.json'  # API endpoint
params = {'start': f'{selected_year}-01-01', 'end': f'{selected_year}-12-31'}  # API parameters
response = requests.get(url, params=params)  # send the request
data = response.json()['bpi']  # convert the response to json and get the data
df = pd.DataFrame.from_dict(data, orient='index', columns=['price'])  # convert the data to a dataframe

# Convert prices to the desired format
df.index = pd.to_datetime(df.index)  # convert the index to datetime
df.index.name = 'date'  # rename the index column to 'date'
df['price'] = df['price'].astype('float')  # convert the price column to float

# Keep only the next 20 days
df_future = df.tail(20)  # get the last 20 rows

# Display Bitcoin prices chart
st.subheader(f'2. Bitcoin Prices chart for the year {selected_year}')  # subheader
st.line_chart(df)  # line chart

# Display prices for the next 20 days
st.subheader('3. Bitcoin Prices for the next 20 days')  # subheader
st.table(df_future)  # table

csv_filename = 'bitcoin_prices.csv'  # name of the CSV file to write to

# Define start and end dates for API request and CSV file
api_start_date = '2018-01-27'  # start date for API request
api_end_date = datetime.date.today().strftime('%Y-%m-%d')  # end date for API request

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
        url = data['next_page']  # get the url for the next page
        response = requests.get(url)  # send the request
        response.raise_for_status()  # raise an exception if the response is not successful
        data = response.json()  # convert the response to json
        for date, price in data['bpi'].items():  # extract date and price data and append to price_data
            price_data.append([date, price])
    except (requests.exceptions.RequestException, KeyError, TypeError) as e:  # catch any errors that may occur
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
except IOError as e:  # catch any errors that may occur
    print('Error while writing data to CSV file:', e)
    exit()

# Load data from CSV file into a dataframe
try:
    with open(csv_filename, mode='r') as csv_file:  # open the CSV file
        bitcoin_df = pd.read_csv(csv_file)  # read the data into a dataframe
except (IOError, pd.errors.EmptyDataError) as e:  # catch any errors that may occur
    print('Error while loading data from CSV file:', e)
    exit()

# Convert 'Price' column to numeric values
bitcoin_df['Price'] = pd.to_numeric(bitcoin_df['Price'], errors='coerce')
# Convert 'Date' column to datetime objects
try:
    bitcoin_df['Date'] = pd.to_datetime(bitcoin_df['Date'])
except ValueError as e:  # catch any errors that may occur
    print('Error while converting data to datetime objects:', e)
    exit()

# Drop rows with missing or invalid data
bitcoin_df.dropna(inplace=True)

# Scale the data
scaler = MinMaxScaler()  # create a scaler object
scaled_data = scaler.fit_transform(bitcoin_df[['Price']])  # fit and transform the data

# Convert the scaled data to a dataframe
bitcoin_df_scaled = pd.DataFrame(scaled_data, columns=['Price'], index=bitcoin_df.index)

# Split data into training and testing sets
train_data = bitcoin_df_scaled[:int(0.8 * len(bitcoin_df))]
test_data = bitcoin_df_scaled[int(0.8 * len(bitcoin_df)):]

# Fit linear regression model
regressor = LinearRegression()  # create a linear regression object
X_train = train_data.index.values.reshape(-1, 1)  # reshape the data
y_train = train_data['Price'].values.reshape(-1, 1)  # reshape the data
regressor.fit(X_train, y_train)  # fit the model

# Make predictions for next 20 days
X_pred = np.arange(len(bitcoin_df), len(bitcoin_df) + 20).reshape(-1, 1)  # create a numpy array for the next 20 days
y_pred_scaled = regressor.predict(X_pred)  # predict the prices for the next 20 days
y_pred = scaler.inverse_transform(y_pred_scaled)  # unscale the predicted prices

# Convert the date range for the next 20 days to datetime objects
dates = pd.date_range(start=bitcoin_df['Date'].max(), periods=20, freq='D')

# Plot the data and predictions
plt.figure(figsize=(12, 6))  # set the figure size
plt.plot(bitcoin_df['Date'], bitcoin_df['Price'], label='Actual')  # plot the actual prices
plt.plot(dates, y_pred, label='Predicted')  # plot the predicted prices
plt.title('Bitcoin Price Prediction')  # set the title
plt.xlabel('Date')  # name the x-axis
plt.ylabel('Price ($)')   # name the y-axis
plt.legend()  # show the legend
plt.show()  # show the plot

# Print predicted price for the next 20 days
print('Predicted prices for the next 20 days:')
print('-' * 10)
for i in range(1, 21):
    print(f'Day {i}: ${y_pred[i - 1][0]:.2f}')

# Plot the Bitcoin price data
sns.set_style("darkgrid")  # set the style to darkgrid
plt.figure(figsize=(12, 6))  # set the figure size
plt.plot(bitcoin_df.index, bitcoin_df['Price'], label='Actual')  # plot the actual prices
plt.title('Bitcoin Price Data', fontsize=16)  # set the title
plt.xlabel('Date', fontsize=14)  # name the x-axis
plt.ylabel('Price (USD)', fontsize=14)  # name the y-axis
plt.legend(fontsize=12)  # show the legend

# Plot the predicted prices for the next 20 days
plt.figure(figsize=(12, 6))  # set the figure size
plt.plot(X_pred, y_pred, label='Predicted')  # plot the predicted prices
plt.title('Bitcoin Price Predictions', fontsize=16)  # set the title
plt.xlabel('Date', fontsize=14)  # name the x-axis
plt.ylabel('Price (USD)', fontsize=14)  # name the y-axis
plt.legend(fontsize=12)   # show the legend

# Show the plots
plt.show()
