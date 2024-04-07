# TradeTracker for Forex Trading

TradeTracker is a sophisticated Python tool designed to automate Forex trading decisions based on predictions from two separate machine learning models trained on negative and positive price actions respectively.

## Overview

The TradeTracker utilizes an `AISignal` class for handling ML model operations and an `OANDATrader` class for interfacing with the OANDA Forex trading platform. It is designed to track daily trades, deciding whether to buy, sell, or hold based on the combined predictions of two LSTM models: one trained on negative price actions ("LeftBrain") and the other on positive price actions ("RightBrain").

## Features

- Automated trading decisions based on ML predictions.
- Distinct models for positive and negative market movements.
- Daily trade tracking with customizable trade duration.
- Integration with OANDA Forex for trade execution.

## Installation

This project requires Python 3.12  and depends on several external libraries including `numpy`, `pandas`, `tensorflow`, and others. It is managed with Poetry for easier dependency handling.

1. **Clone the repository:**


2. **Install dependencies using Poetry:**

If you haven't installed Poetry, refer to the [official Poetry documentation](https://python-poetry.org/docs/#installation) for guidance.


3. **Activate the virtual environment:**


## Usage

To utilize TradeTracker, ensure you have both "LeftBrain" and "RightBrain" models trained and saved within the `Model` directory.

1. **Initialize TradeTracker:**

Instantiate the `TradeTracker` class. Ensure your OANDA credentials (`OANDA_ID` and `OANDA_TOKEN`) are correctly set in `configForex.py`.

```python
from TradeTracker import TradeTracker

tracker = TradeTracker()

tracker.track_daily_trades(num_days=80, model_left="LeftBrain (Negative).h5", model_right="RightBrain (Positive).h5")

trade_data_df = tracker.get_trade_data()
