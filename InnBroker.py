# For Interactive Brokers
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper
from ibapi.client import EClient


# Interactive Brokers API class
class InteractiveBrokersAPI(EWrapper, EClient):
    def __init__(self, ai_model):
        EClient.__init__(self, self)
        self.connect("127.0.0.1", 7497, 0)  # default TWS paper trading port
        self.ai_model = ai_model  # Composition

    def execute_trades(self, symbol):
        signal = self.ai_model.get_signal(symbol)
        if signal == 'buy':
            self.place_order(symbol, 1, 'BUY')
        elif signal == 'sell':
            self.place_order(symbol, 1, 'SELL')
        elif signal == 'hold':
            pass

    # (Mockup for placing an order)
    def place_order(self, symbol, qty, action):
        print(f"Placing order: {symbol}, {qty}, {action}")