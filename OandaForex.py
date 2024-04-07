import math
import asyncio
import oandapyV20
import configForex
import pandas as pd
from datetime import datetime, timedelta
import oandapyV20.endpoints.orders as orders
from TradingIndicators import StockIndicators
import oandapyV20.endpoints.pricing as pricing
from sklearn.preprocessing import StandardScaler
import oandapyV20.endpoints.accounts as accounts
from oandapyV20.endpoints.pricing import PricingInfo
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.positions as positions
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails



class OANDATrader:
    def __init__(self, account_id, access_token):
        self.client = oandapyV20.API(access_token=access_token)
        self.account_id = account_id

    def get_account_balance(self):
        """Retrieve the account balance."""
        account_summary = accounts.AccountSummary(self.account_id)
        response = self.client.request(account_summary)
        balance = float(response['account']['balance'])
        return balance
    def fetch_current_price(self,instrument):
        params = {"instruments": instrument}
        r = PricingInfo(accountID=self.account_id, params=params)
        rv = self.client.request(r)
        price = float(rv['prices'][0]['closeoutBid'])
        return price
    def create_order_with_risk_management(self, instrument, is_buy, pip_size=0.002, risk_percentage=0.03, reward_to_risk_ratio=3):
        """Create a market order with risk management and 3:1 reward to risk ratio."""
        balance = self.get_account_balance()
        amount_to_risk = balance * risk_percentage
        entry_price = self.fetch_current_price(instrument=instrument)
        ratio = (entry_price * pip_size)
        # Set stop loss and take profit based on the entry price and whether it's a buy or sell order
        if is_buy:
            profit_price = entry_price + (ratio * reward_to_risk_ratio)
            loss_price = entry_price - (ratio)
            units = math.floor(amount_to_risk / loss_price)
        else:
            profit_price = entry_price - (ratio * reward_to_risk_ratio)
            loss_price = entry_price + (ratio)
            units = -(math.floor(amount_to_risk / loss_price))
        self.create_order(instrument, units, str(profit_price), str(loss_price))
    def create_order(self, instrument, units, take_profit=None, stop_loss=None):
        """Create a market order with optional take profit and stop loss orders."""
        # Prepare the details for take profit and stop loss if provided
        tp_details = TakeProfitDetails(price=take_profit).data if take_profit else None
        sl_details = StopLossDetails(price=stop_loss).data if stop_loss else None
        # Create the market order request with the necessary details
        mktOrder = MarketOrderRequest(
            instrument=instrument,
            units=units,
            takeProfitOnFill=tp_details,
            stopLossOnFill=sl_details
        )
        # Use the request to create an order
        order = orders.OrderCreate(self.account_id, data=mktOrder.data)  # Note the use of .data here
        response = self.client.request(order)
        print(response)
    def fetch_market_data(self, instrument, params=None):
        # Define the end time as now and start time as 6 months before
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=365*3)  # Approximating 4 years
        params = {
            "granularity": "D",  # Daily candles
            "from": start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "to": end_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        }
        # Create and send the request to get candlestick data
        request = instruments.InstrumentsCandles(instrument=instrument, params=params)
        response = self.client.request(request)
        # Parse the response to structure it into a DataFrame
        data = []
        for candle in response['candles']:
            time = candle['time']
            volume = candle['volume']
            complete = candle['complete']
            o = float(candle['mid']['o'])
            h = float(candle['mid']['h'])
            l = float(candle['mid']['l'])
            c = float(candle['mid']['c'])
            data.append([time, o, h, l, c, volume, complete])
        # Create a DataFrame
        df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'complete'])
        # Convert the 'Time' column to a datetime format if desired
        df['time'] = pd.to_datetime(df['time'])
        indicator = StockIndicators(df)
        data = indicator.append().dropna()
        return data
    def get_account_summary(self):
        """Get a summary of the account."""
        summary = accounts.AccountSummary(self.account_id)
        response = self.client.request(summary)
        print(response)
    def remove_specified_columns(self, dataframe, columns_to_remove):
        # Convert all column names in the DataFrame and the columns_to_remove list to lowercase
        dataframe.columns = dataframe.columns.str.upper()
        columns_to_remove = [column.upper() for column in columns_to_remove]
    
        return dataframe.drop(columns=columns_to_remove, errors='ignore')
    def has_open_positions(self, instrument):
        """
        Check if there are any open positions for the account.

        Returns:
            bool: True if there are open positions, False otherwise.
        """
        request = positions.PositionDetails(accountID=self.account_id, instrument=instrument)
        response = self.client.request(request)
        filtered_response = {
            key: value for key, value in response.items()
            if (isinstance(value, dict) and
                (('long' in value and value['long']['units'] != '0') or
                ('short' in value and value['short']['units'] != '0')))
        }
        if len(filtered_response) < 1:
            filtered_response = None
        return filtered_response

class DataFrameAnalysis:
    def __init__(self, dataframe):
        self.df = dataframe

    def set_target(self):
        # Define bullish conditions
        bullish_conditions = (
            (self.df['close'] > self.df['SMA']) &
            (self.df['close'].shift(1) <= self.df['Bollinger_Upper'].shift(1)) &
            (self.df['close'] > self.df['Support']) &
            (self.df['close'] > self.df['VWAP']) &
            (self.df['RSI'] > 40)
        )

        # Define bearish conditions
        bearish_conditions = (
            (self.df['close'] < self.df['SMA']) &
            (self.df['close'].shift(1) >= self.df['Bollinger_Lower'].shift(1)) &
            (self.df['close'] < self.df['Resistance']) &
            (self.df['close'] < self.df['VWAP']) &
            (self.df['RSI'] <= 40)
        )

        # Initialize the 'target' column with zeros
        self.df['target'] = 0

        # Set the target to 1 where bullish conditions are met
        self.df.loc[bullish_conditions, 'target'] = 1

        # Set the target to -1 where bearish conditions are met
        self.df.loc[bearish_conditions, 'target'] = -1

        return self.df
    def normalize_features(self, dataframe):
        # Separate the timestamp column
        timestamp_data = dataframe['TIMESTAMP']
        data_to_normalize = dataframe.drop('TIMESTAMP', axis=1)
        # Normalize the data
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data_to_normalize)
        # Convert normalized data back to DataFrame
        normalized_df = pd.DataFrame(normalized_data, columns=data_to_normalize.columns)
        # Reattach the timestamp column
        normalized_df = pd.concat([timestamp_data.reset_index(drop=True), normalized_df], axis=1)
        return normalized_df

class MarketAnalysisAndOrder:
    def __init__(self, instrument='EUR_USD'):
        self.trader = OANDATrader(configForex.OANDA_ID, configForex.OANDA_TOKEN)
        self.instrument = instrument
        self.fetch_and_process_data()
        self.normalize_data()

    def fetch_and_process_data(self):
        self.data = self.trader.fetch_market_data(self.instrument)  # Fetch the most recent data
        df_analysis = DataFrameAnalysis(self.data)
    
        # Set targets based on bullish and bearish conditions
        self.data = df_analysis.set_target().rename(columns={'time': 'timestamp'})
    
        columns_to_remove = ['complete']
        self.data = self.trader.remove_specified_columns(self.data, columns_to_remove)

    def normalize_data(self):
        df_analysis = DataFrameAnalysis(self.data)
        self.normalized_data = df_analysis.normalize_features(self.data.drop('TARGET', axis=1))
        self.normalized_data['TARGET'] = self.data['TARGET'].values  # Ensure 'TARGET' is in self.normalized_data

    def split_features_and_target(self):
        features = self.normalized_data.drop('TARGET', axis=1)
        target = self.normalized_data['TARGET']
        return features, target

class OandaStreamer:
    def __init__(self):
        self.account_id = configForex.OANDA_ID
        self.client = oandapyV20.API(access_token=configForex.OANDA_TOKEN)
        self.price_updates_df = pd.DataFrame(columns=['instrument', 'time', 'bid_price', 'bid_liquidity', 'ask_price', 'ask_liquidity', 'closeoutBid', 'closeoutAsk', 'status', 'tradeable'])
        self.stream_active = False

    def start_streaming(self):
        """Start the streaming connection and handle updates."""
        request_args = {
            "instruments": "EUR_USD,GBP_USD,AUD_USD",  # Replace with the desired instruments
            "snapshot": True
        }
        stream = pricing.PricingStream(accountID=self.account_id, params=request_args)
        self.stream_active = True
        r = self.client.request(stream)

        try:
            for update in r:
                if update['type'] == 'PRICE':
                    self.handle_price_update(update)
                if not self.stream_active:
                    break
        except Exception as e:
            print(f"Error occurred during streaming: {e}")
        finally:
            stream.terminate()

    def handle_price_update(self, update):
        """Handle price updates."""
        instrument = update['instrument']
        time = update['time']
        bid_price = update['bids'][0]['price']
        bid_liquidity = update['bids'][0]['liquidity']
        ask_price = update['asks'][0]['price']
        ask_liquidity = update['asks'][0]['liquidity']
        closeout_bid = update['closeoutBid']
        closeout_ask = update['closeoutAsk']
        status = update['status']
        is_tradeable = update['tradeable']

        # Store the price update in the DataFrame
        price_update = {
            'instrument': instrument,
            'time': time,
            'bid_price': bid_price,
            'bid_liquidity': bid_liquidity,
            'ask_price': ask_price,
            'ask_liquidity': ask_liquidity,
            'closeoutBid': closeout_bid,
            'closeoutAsk': closeout_ask,
            'status': status,
            'tradeable': is_tradeable
        }
        self.price_updates_df = self.price_updates_df._append(price_update, ignore_index=True)

        # Print the price update
        print(f"Price update for {instrument}:")
        print(f"  Time: {time}")
        print(f"  Bid Price: {bid_price}")
        print(f"  Bid Liquidity: {bid_liquidity}")
        print(f"  Ask Price: {ask_price}")
        print(f"  Ask Liquidity: {ask_liquidity}")
        print(f"  Closeout Bid: {closeout_bid}")
        print(f"  Closeout Ask: {closeout_ask}")
        print(f"  Status: {status}")
        print(f"  Tradeable: {is_tradeable}")

    def stop_streaming(self):
        """Stop the streaming connection."""
        self.stream_active = False

    def get_price_updates_df(self):
        """Get the price updates DataFrame."""
        return self.price_updates_df