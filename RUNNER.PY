from AiSignaler import AISignal
from CustomException import *
from OandaForex import *
import datetime, time
import pandas as pd
import numpy as np

class TradeTracker:
    def __init__(self):
        self.model = AISignal()
        self.trader = OANDATrader(configForex.OANDA_ID, configForex.OANDA_TOKEN)
        self.columns = ['Date', 'Time', 'Models', 'Trade Number', 'Balance', 'Instrument', 'FILEPATH', 'FILEPATH2', 'Units', 'marginUsed', 'averagePrice']
        self.data = []

    def track_daily_trades(self, num_days, model_left, model_right, instrument="EUR_USD"):
        start_date = datetime.date.today()
        end_date = start_date + datetime.timedelta(days=num_days)

        trade_num = 0
        while True:
            try:
                current_date = datetime.date.today()
                current_time = datetime.datetime.now()
                print(f"Current date and time: {current_date} {current_time}")

                if current_date >= end_date:
                    print(f"Loop ended on {current_date}")
                    break

                try:
                # logic to get the number of trades, balance, and instrument
                    predictions_left = self.create_sequences_and_predict(model_left, instrument)
                    predictions_right = self.create_sequences_and_predict(model_right, instrument)
                    response = self.trader.has_open_positions(instrument)
                    print(f"prediciton left: {predictions_left} prediction right: {predictions_right}"  )
                except Exception as e:
                    self._handle_exception(e, "Error occurred while getting predictions or open positions")
                    continue

                try:
                    if sum(predictions_left) < sum(predictions_right) and (response == None):
                        # Buy signal
                        self.trader.create_order_with_risk_management(instrument, True)
                        response = self.trader.has_open_positions(instrument)
                        trade_num += 1
                        balance = self.trader.get_account_balance()
                        device = instrument
                        file_path = self.model.get_model_file_path(model_left)
                        file_path2 = self.model.get_model_file_path(model_right)
                        marginUsed = response['position']['marginUsed']
                        units = response['position']['long']['units']
                        averagePrice = response['position']['long']['averagePrice']
                        print("BUY")
                        # Get the current time
                        current_time_str = current_time.strftime("%H:%M:%S")
                        # Append data to the list
                        self.data.append([
                            current_date,
                            current_time_str,
                            model_left,
                            trade_num,
                            balance,
                            device,
                            file_path,
                            file_path2,
                            units,
                            marginUsed,
                            averagePrice
                        ])
                    elif (sum(predictions_left) == sum(predictions_right)) or response:
                        # Hold signal
                        # Do nothing or add any desired logic for holding
                        print("Hold")
                        while response:
                            try:
                                # Wait for a certain interval before checking again
                                time.sleep(300)  # Wait for 5 min, adjust as needed
                                current_date = datetime.date.today()
                                current_time = datetime.datetime.now()
                                print(f"Current date and time: {current_date} {current_time}")
                                response = self.trader.has_open_positions(instrument)
                            except Exception as e:
                                self._handle_exception(e, "Error occurred while waiting for open positions to close")
                                break
                    elif (response == None):
                        # Sell signal
                        self.trader.create_order_with_risk_management(instrument, False)
                        response = self.trader.has_open_positions(instrument)
                        trade_num += 1
                        balance = self.trader.get_account_balance()
                        device = instrument
                        file_path = self.model.get_model_file_path(model_left)
                        file_path2 = self.model.get_model_file_path(model_right)
                        marginUsed = response['position']['marginUsed']
                        units = response['position']['short']['units']
                        averagePrice = response['position']['short']['averagePrice']
                        print("SELL")
                        # Get the current time
                        current_time_str = current_time.strftime("%H:%M:%S")
                        # Append data to the list
                        self.data.append([
                            current_date,
                            current_time_str,
                            model_right,
                            trade_num,
                            balance,
                            device,
                            file_path,
                            file_path2,
                            units,
                            marginUsed,
                            averagePrice
                        ])
                        # Add any additional logic for selling, if needed
                except Exception as e:
                    self._handle_exception(e, "Error occurred while processing signals")
                    continue

                try:
                    # Get the current day of the week (0 = Monday, 1 = Tuesday, ..., 6 = Sunday)
                    current_day = datetime.datetime.today().weekday()

                    # Check if the current day is a Saturday (5) or Sunday (6)
                    if current_day == 5 or current_day == 6:
                        # Calculate the number of seconds until Monday
                        if current_day == 5:  # Saturday
                            sleep_duration = (24 - datetime.datetime.now().hour) * 3600 + (60 - datetime.datetime.now().minute) * 60 + (60 - datetime.datetime.now().second)
                        else:  # Sunday
                            sleep_duration = (48 - datetime.datetime.now().hour) * 3600 + (60 - datetime.datetime.now().minute) * 60 + (60 - datetime.datetime.now().second)
                        time.sleep(sleep_duration)
                except Exception as e:
                    self._handle_exception(e, "Error occurred while handling weekends")
                    continue

                time.sleep(60)
            except Exception as e:
                self._handle_exception(e, "Error occurred in the main loop")
                continue
    def get_trade_data(self):
        # Create a DataFrame from the data
        df = pd.DataFrame(self.data, columns=self.columns)
        return df
    def create_sequences_and_predict(self, model_name, instrument):
        def create_sequences(data, sequence_length):
            sequences = []
            for i in range(len(data) - sequence_length + 1):
                sequence = data[i:i+sequence_length]
                sequences.append(sequence)
            return np.array(sequences)
        # Load the trained model
        self.model.load_model(model_name)
        # Fetch data from Oanda
        market_analysis = MarketAnalysisAndOrder(instrument=instrument)
        normalized_df = market_analysis.normalized_data
        features = normalized_df.drop(['TARGET', 'TIMESTAMP', 'AVG_VOLUME', 'BOLLINGER_MIDDLE', 'OPEN', 'HIGH'], axis=1).values
        # Create input sequences
        sequence_length = 30
        prediction_sequences = create_sequences(features, sequence_length)
        # Check the shape of prediction_sequences
        print("Shape of prediction_sequences:", prediction_sequences.shape)
        # Reshape prediction_sequences if needed
        expected_shape = (len(prediction_sequences), sequence_length, features.shape[1])
        if prediction_sequences.shape != expected_shape:
            prediction_sequences = prediction_sequences.reshape(expected_shape)
        # Get predictions
        predicted_classes = self.model.get_signal(prediction_sequences)
        return predicted_classes[:10]
    def _handle_exception(self, e, line_info=""):
        logging.error(f"Caught an exception [ RUNNER ]: {e} {line_info}")

tracker = TradeTracker()
tracker.track_daily_trades(80, "LeftBrain (Negative).h5", "RightBrain (Positive).h5")