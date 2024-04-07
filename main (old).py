from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from AlpacaClient import AlpacaHistoricalData
from TradingIndicators import StockIndicators
from AlpacaClient import AlpacaAPI
from AiSignaler import AISignal
from multiprocessing import Pool
from OandaForex import *
import multiprocessing
import yfinance as yf
import pandas as pd
import numpy as np
import os

def normalize_features(dataframe):
    # Separate the timestamp column
    timestamp_data = dataframe['timestamp']
    data_to_normalize = dataframe.drop('timestamp', axis=1)
    # Normalize the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data_to_normalize)
    # Convert normalized data back to DataFrame
    normalized_df = pd.DataFrame(normalized_data, columns=data_to_normalize.columns)
    # Reattach the timestamp column
    normalized_df = pd.concat([timestamp_data.reset_index(drop=True), normalized_df], axis=1)
    return normalized_df

def create_binary_target(dataframe, price_column):
    # Separate the timestamp column
    timestamp_data = dataframe['timestamp']
    data_to_normalize = dataframe.drop('timestamp', axis=1)
    # Assuming 'price_column' is the name of the column with stock prices
    # Shift the price column to compare today's price with the next day's
    shifted_prices = dataframe[price_column].shift(-1)
    # Binary target: 1 if next day's price is higher than today's, 0 otherwise
    dataframe['target'] = (shifted_prices > dataframe[price_column]).astype(int)
    return dataframe

def prepare_data_for_tensorflow(df, timestamp_cols=None):
    # If there are timestamp columns, we'll exclude them from scaling
    if timestamp_cols is not None:
        timestamp_data = df[timestamp_cols]
        data_to_scale = df.drop(columns=timestamp_cols)
    else:
        data_to_scale = df
    # Initialize the StandardScaler
    scaler = StandardScaler()
    # Scale the data excluding timestamp columns
    scaled_data = scaler.fit_transform(data_to_scale)
    # Convert scaled data back to DataFrame
    scaled_data_df = pd.DataFrame(scaled_data, columns=data_to_scale.columns)
    # If there were timestamp columns, add them back to the DataFrame
    if timestamp_cols is not None:
        scaled_data_df = pd.concat([timestamp_data.reset_index(drop=True), scaled_data_df.reset_index(drop=True)], axis=1)
    return scaled_data_df


def create_3d_sequences(features, target, timesteps, test_size=0.2, random_state=42):
    """
    Split the data into train and test sets and convert features to 3D format for LSTM.
    :param features: DataFrame containing the features.
    :param target: Series or DataFrame column containing the target.
    :param timesteps: Number of timesteps for each sequence.
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_state: Controls the shuffling applied to the data before applying the split.
    :return: X_train, X_test, y_train, y_test where feature sets are in 3D format.
    """
    # Split data (2D structure)
    X_train_2d, X_test_2d, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)
    # Function to create 3D sequences from 2D data
    def create_sequences_2d_to_3d(data_2d, target, timesteps):
        sequences = []
        target_sequence = []
        for i in range(len(data_2d) - timesteps):
            sequences.append(data_2d.iloc[i:(i + timesteps)].values)
            target_sequence.append(target.iloc[i + timesteps])
        return np.array(sequences), np.array(target_sequence)
    # Transform the 2D train and test feature sets to 3D
    X_train_3d, y_train = create_sequences_2d_to_3d(X_train_2d, y_train, timesteps)
    X_test_3d, y_test = create_sequences_2d_to_3d(X_test_2d, y_test, timesteps)
    return X_train_3d, X_test_3d, y_train, y_test

def create_combonations(features):
    """
    Creates all possiblites/combinations of a list and returns the all possible combinations
    """
    import itertools
    max_combination_size = 3  # You can adjust this
    all_combinations = []
    for r in range(2, max_combination_size + 1):
        combinations_r = itertools.combinations(features, r)
        all_combinations.extend(combinations_r)
    return all_combinations

def alpaca_prep(stock, symbol):
    test = AlpacaAPI()
    test.connect_account()
    ClientData = AlpacaHistoricalData()
    data = ClientData.get_historical_data(stock, symbol)
    indicator = StockIndicators(data.df)
    data = indicator.append().dropna()
    # Example DataFrame
    df = pd.DataFrame(data)
    # Assuming your DataFrame is named df
    df = df.reset_index()
    # If you want to name the new column that contains the datetime information:
    df = df.reset_index().rename(columns={'index': 'timestamp'})
    # Select features and target
    features = normalize_features(df[['timestamp','open', 'high', 'low', 'volume', 'SMA', 'Support', 'Resistance']])
    features = features.iloc[:, 2:]
    features = prepare_data_for_tensorflow(features)
    target = create_binary_target(df ,'close')
    return features, target

class StockData:
    def __init__(self, ticker):
        self.ticker = ticker
    def get_data(self, start_date, end_date):
        """
        Fetches stock data for the specified date range.
        
        :param start_date: Start date in 'YYYY-MM-DD' format.
        :param end_date: End date in 'YYYY-MM-DD' format.
        :return: Pandas DataFrame with stock data.
        """
        data = yf.download(self.ticker, start=start_date, end=end_date)
        data.rename(columns={
            'Open': 'open',
            'Close': 'close',
            'High': 'high',
            'Low': 'low',
            'Adj Close': 'target',
            'Volume': 'volume'
        }, inplace=True)
        data = pd.DataFrame(data)
        data = data.reset_index()
        data = data.rename(columns={'Date': 'timestamp'})
        indicator = StockIndicators(data)
        data = indicator.append().dropna()
        data = normalize_features(data)
        features = (data.drop('target', axis=1)).drop(columns="timestamp")
        target = create_binary_target(data,'close')
        return features, target

def train_and_evaluate(feature, target, combinations):
    columns_to_remove = list(combinations)
    features = feature.drop(columns=columns_to_remove, axis=1, errors='ignore')
    
    # Split data (3D structure)
    timesteps = 30  # Set your timesteps here
    X_train, X_test, y_train, y_test = create_3d_sequences(features, target['TARGET'], timesteps)
    # Now, train your model (assuming AISignal is correctly implemented)
    # Example configuration
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    hidden_layers_config = [
        {'units': 64, 'activation': 'relu', 'return_sequences': True},  # First LSTM layer
        {'units': 32, 'activation': 'relu'} # Third LSTM layer, adjusted 'return_sequences'
        # Additional layers can be added here in the same manner
    ]
    # Initialize the AISignal class with the hidden layers configuration
    model = AISignal(input_shape=X_train.shape[-2:], hidden_layers_config=hidden_layers_config)
    if os.path.exists(current_dir + "\Model"):
        model.load_model(current_dir + "AlpacaTrainer 0.0.0.1v")
        model.train(X_train, y_train)
    # Evaluate your model
    result = model.evaluate(X_test, y_test)
    model.save_model("AlpacaTrainer 0.0.0.1v")
    return combinations, result

def remove_specified_columns(dataframe, columns_to_remove):
    # Convert all column names in the DataFrame and the columns_to_remove list to lowercase
    dataframe.columns = dataframe.columns.str.lower()
    columns_to_remove = [column.lower() for column in columns_to_remove]
    
    return dataframe.drop(columns=columns_to_remove, errors='ignore')

if __name__ == '__main__':
    analysis = MarketAnalysisAndOrder().normalized_data.drop(['TARGET', 'TIMESTAMP'], axis=1)
    feature = analysis.normalized_data
    features = feature.drop(['TARGET', 'TIMESTAMP'], axis=1)
    target = analysis.normalized_data
    target = target.drop('TIMESTAMP', axis=1)
    combo_config = ['BOLLINGER_MIDDLE',
                    'BOLLINGER_LOWER',
                    'BOLLINGER_UPPER',
                    'RESISTANCE',
                    'AVG_VOLUME',
                    'SUPPORT',
                    'VOLUME',
                    'CLOSE',
                    'VWAP',
                    'HIGH',
                    'OPEN',
                    'LOW',
                    'RSI',
                    'SMA',
                    ]
    all_combinations = create_combonations(combo_config)
    args_for_pool = [(features, target, combination) for combination in all_combinations]
    pool = Pool(processes=(multiprocessing.cpu_count() - 4))  # or any number of processes you wish
    results = pool.starmap(train_and_evaluate, args_for_pool)
    pool.close()
    pool.join()

    my_dict = dict(results)
    df = pd.DataFrame.from_dict(my_dict, orient='index').reset_index()
    custom_headers = ["Feature Combination", "Loss", "MAE"]
    df.columns = custom_headers
    chunk_size = 25  # Define a suitable chunk size based on your data and system capabilities
    with pd.ExcelWriter('output.xlsx', engine='openpyxl', mode='w') as writer:
        for start in range(0, df.shape[0], chunk_size):
            end = min(start + chunk_size, df.shape[0])
            chunk = df.iloc[start:end]
            chunk.to_excel(writer, startrow=start, index=False, header=["Feature Combination", "Loss", "MAE"])
    #df.to_excel('output.xlsx', index=False, header=["Feature Combination", "Loss", "MAE"])


'''

plt.plot(timestamps, data['SMA'], label='SMA')  
plt.plot(timestamps, data['open'], label='Open')
plt.plot(timestamps, data['Support'], label='Support')
plt.plot(timestamps, data['Resistance'], label='Resistance')
plt.plot(timestamps, data['Bollinger_Upper'], label='Bollinger Upper')
plt.plot(timestamps, data['Bollinger_Lower'], label='Bollinger Lower')
plt.plot(timestamps, data['VWAP'], label='VWAP') 
plt.title('Trendlines of TSLA Stock')
plt.xlabel('Timestamp')
plt.ylabel('Price')
plt.legend()
plt.show()

'''