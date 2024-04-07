import alpaca_trade_api as TradingAPI
import requests
import config

ORDERS_URL = '{}/v2/orders'.format(config.APCA_API_BASE_URL)

def createMarketOrder():

    ticker = 'AAPL'
    qty = '2'
    side = 'buy'
    ordertype = 'market'
    
    data = {
        'symbol': ticker,
        'qty': qty,
        'side': side,
        'type': ordertype,
        'time_in_force': 'day'
    }
    r = requests.post(ORDERS_URL, json=data, headers=config.HEADERS)
    return r.json()

def get_account_info():
    ACCOUNT_URL = f"{config.APCA_API_BASE_URL}/v2/account"
    response = requests.get(ACCOUNT_URL, headers=config.HEADERS)

    if response.status_code == 200:  # HTTP 200 means OK
        account_info = response.json()
        return account_info
    else:
        print(f"Failed to get account information. Error: {response.content}")
        return None

print(createMarketOrder())
