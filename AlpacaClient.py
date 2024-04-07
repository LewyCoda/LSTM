# For Alpaca
from requests.exceptions import ConnectionError, Timeout, RequestException
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import alpaca_trade_api as tradeapi
import requests
import config

# Exception Handling Custom
from CustomException import *

# Alpaca API class ai_model,base_url='https://paper-api.alpaca.markets'
class APIRequestHandler:
    def __init__(self, api_base_url, headers):
        self.api_base_url = api_base_url
        self.headers = headers
    
    def _handle_exception(self, e, line_info=""):
        logging.error(f"Exception [ APIRequestHndler ]: {e} {line_info}")

    def make_request(self, url):
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self._handle_exception(e, "Line 58")  # Raise exception 

class AlpacaDataFetcher:
    def __init__(self, api_request_handler):
        self.api_request_handler = api_request_handler

    def _handle_exception(self, e, line_info=""):
        logging.error(f"Exception [ AlpacaDataFetcher ]: {e} {line_info}")
        
    def fetch_stock_data(self, symbol, endpoint="/v2/assets"):
        url = f"{self.api_request_handler.api_base_url}{endpoint}/{symbol}"
        return self.api_request_handler.make_request(url)

class AlpacaAPI:
    def __init__(self):
        self.api_request_handler = APIRequestHandler(api_base_url="https://paper-api.alpaca.markets", headers=config.HEADERS)
        self.data_fetcher = AlpacaDataFetcher(self.api_request_handler)
        self.account_info = None
        self.stock = None
        self.orders = []

    def _handle_exception(self, e, line_info=""):
        logging.error(f"Caught an exception [ AlpacaAPI ]: {e} {line_info}")

    def get_stock(self, symbol):
        try:
            self.stock = self.data_fetcher.fetch_stock_data(symbol)
            return self.display_info("Stock", self.stock)
        except Exception as e:
            self._handle_exception(e, "LINE X")

    def get_orders(self):
        orders_url = f"{self.api_request_handler.api_base_url}/v2/orders"
        try:
            response = requests.get(orders_url, headers=self.api_request_handler.headers)
            if response.status_code == 200:
                self.display_info("Order", response.json)
            else:
                print(f"Failed to place order. {response.content} Endpoint: {orders_url}")
        except Exception as e:
            self._handle_exception(e, f"Failed to submit order for {self.order['symbol']}")

    def connect_account(self):
        account_url = f"{self.api_request_handler.api_base_url}/v2/account"
        try:
            self.account_info = self.api_request_handler.make_request(account_url)
            self.display_info("Account", self.account_info)
        except Exception as e:
            self._handle_exception(e, "LINE X")

    def execute_trades(self, symbol):
        signal = self.ai_model.get_signal(symbol)
        if "signal" == 'buy':
            self.place_order(symbol, 1, 'buy', 'limit', 'gtc')
        elif "signal" == 'sell':
            self.place_order(symbol, 1, 'sell', 'limit', 'gtc')
        elif "signal" == 'hold':
            pass
    
    def set_symbol(self, symbol):
        self.symbol = symbol

    def set_orders(self, symbol, qty, side, type, time_in_force):
        order = {
            'symbol': symbol,
            'qty': qty,
            'side': side,
            'type': type,
            'time_in_force': time_in_force
        }
        self.orders.append(order)
        
    def submit_orders(self):
        if not self.orders:
            print("No orders to submit")
            return
        orders_url = f"{self.api_request_handler.api_base_url}/v2/orders"
        for order in self.orders:
            try:
                response = requests.post(orders_url, json=order, headers=self.api_request_handler.headers)
                if response.status_code == 200:
                    print(f"Order for {order['symbol']} submitted successfully.")
                else:
                    print(f"Failed to place order. {response.content} Endpoint: {orders_url}")
            except Exception as e:
                self._handle_exception(e, f"Failed to submit order for {order['symbol']}")

    def display_info(self, info_type, info_dict ):
        if self.account_info or self.stock:
                print(f"{info_type} Information:")
                print("-" * 40)
                for key, value in info_dict.items():
                    print(f"{key}: {value}")
                print("-" * 40)
        else:
            print("No account information stored.")

    def display_stock(self):
        if self.stock:
            self.display_info("Stock", self.stock)
        else:
            print("Run [get_stock()]")

    def display_account(self):
        self.display_info("Account", self.account_info)

class AlpacaHistoricalData:
    def __init__(self, api_key=config.ALPACA_KEY, api_secret=config.ALPACA_SECRET_KEY, base_url=config.APCA_API_BASE_URL):
        self.config = {
            'crypto': {
                'client': CryptoHistoricalDataClient(),
                'months': 18,
                'timeframe': TimeFrame.Day
            },
            'stock': {
                'client': StockHistoricalDataClient(api_key, api_secret),
                'months': 18,
                'timeframe': TimeFrame.Day
            }
        }
    def _handle_exception(self, e, line_info=""):
        logging.error(f"Caught an exception [ AlpacaHistoricalData ]: {e} {line_info}")

    def set_timeframe(self, market_type, months):
        if market_type in self.config:
            self.config[market_type]['months'] = months
        else:
            print(f"Unknown market type: {market_type}")

    def get_historical_data(self, market_type, symbols):
        if market_type not in self.config:
            print(f"Unknown market type: {market_type}")
            return None


        end_date = datetime.now()
        start_date = end_date - relativedelta(months=self.config[market_type]['months']) 

        try:
            if market_type == 'crypto':
                request_params = CryptoBarsRequest(
                                    symbol_or_symbols=symbols,
                                    timeframe=self.config['crypto']['timeframe'],
                                    start=start_date,
                                    end=end_date
                                )

                response = self.config['crypto']['client'].get_crypto_bars(request_params)

            elif market_type == 'stock':
                end_date = end_date - timedelta(hours=10)
                request_params = StockBarsRequest(
                                symbol_or_symbols=symbols,
                                timeframe=TimeFrame.Day,#self.config['stock']['timeframe'],
                                start=start_date,
                                end=end_date
                            )
                print(f"Request Parameters:")
                print(f"  symbol_or_symbols: {symbols}")
                print(f"  timeframe: {self.config['stock']['timeframe']}")
                print(f"  start_date: {start_date}")
                print(f"  end_date: {end_date}")
                response = self.config['stock']['client'].get_stock_bars(request_params)
            return response

        except Exception as e:
            self._handle_exception(e, "LINE 196")
        




















#    # Your code here
#    # Example usage:
#    input_shape = (20,) #shape your input data has
#
#    # number of neurons in hidden layers
#    hidden_layers_config = [
#        {'units': 128, 'activation': 'relu'},
#        {'units': 64, 'activation': 'relu', 'kernel_regularizer': tf.keras.regularizers.l1(0.01)}
#    ]
#    output_shape = 2 # number of output neurons
#
#    # Initialize the AI model
#    ai_model1 = AISignaler(input_shape=input_shape, hidden_layers_config=hidden_layers_config, output_shape=output_shape)
#    #ai_model2 = AISignaler(input_shape=input_shape, hidden_layers_config=hidden_layers_config, output_shape=output_shape)
#
#    # Initialize the Alpaca API with the AI model
#    alpaca = AlpacaAPI(ai_model1, api_alpaca_key, api_alpaca_secret)
#    #alpaca.execute_trades("AAPL")
    


# Initialize the Interactive Brokers API with the AI model
# Note: Before running, ensure that Trader Workstation (TWS) is running and the API port is configured correctly.
#ib = InteractiveBrokersAPI(ai_model)
#ib.run()  # This will keep running; you might want to put it in a separate thread.
#ib.execute_trades("AAPL")