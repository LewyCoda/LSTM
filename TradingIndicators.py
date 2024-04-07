import logging

class StockIndicators:

    def __init__(self, data, window_size = 30):
        self.window = window_size
        self.data = data
        self.sma = SMA(self.data)
        self.bollinger = BollingerBands(self.data, self.sma)
        self.sarl = SupportResistanceLevels(self.data)
        self.volume = Volume(self.data)
        self.vwap = VWAP(self.data)
        self.rsi = RSI(self.data)
    
    def display(self):
        """
        Displays key statistics from each indicator calculation.
        """
        print("Simple Moving Average:", self.sma.calculate(self.window).tail())
        lower, middle, upper = self.bollinger.calculate(self.window)
        print(f"Bollinger Bands:\n Lower: {lower.tail()}\n Middle: {middle.tail()}\n Upper: {upper.tail()}")
        support, resistance = self.sarl.calculate_support_resistance(self.window)
        print(f"Support Levels: {support.tail()}\nResistance Levels: {resistance.tail()}")
        print("Average Volume:", self.volume.average_volume(self.window).tail())
        print("VWAP:", self.vwap.calculate_vmap(self.window).tail())
        print("RSI:", self.rsi.calculate_rsi(self.window).tail())

    def append(self):
        # Calculate SMA and append to the main data
        sma_df = self.sma.calculate(self.window)
        self.data['SMA'] = sma_df.values
    
        # Calculate Bollinger Bands and append to the main data
        lower, middle, upper = self.bollinger.calculate(self.window)
        self.data['Bollinger_Lower'] = lower.values
        self.data['Bollinger_Middle'] = middle.values
        self.data['Bollinger_Upper'] = upper.values
    
        # Calculate Support and Resistance Levels and append to the main data
        support, resistance = self.sarl.calculate_support_resistance(self.window)
        self.data['Support'] = support.values
        self.data['Resistance'] = resistance.values
    
        # Calculate Volume and append to the main data
        volume_df = self.volume.average_volume(self.window)
        self.data['Avg_Volume'] = volume_df.values
    
        # Calculate VWAP and append to the main data
        vwap_df = self.vwap.calculate_vmap(self.window)
        self.data['VWAP'] = vwap_df.values
    
        # Calculate RSI and append to the main data
        rsi_df = self.rsi.calculate_rsi(self.window)
        self.data['RSI'] = rsi_df.values
        
        return self.data

class SMA:
    """
    Simple Moving Average (SMA) calculator.
    :param data: The stock data in the form of a Pandas DataFrame.

    !! Window size is dependent on time frame. !!
    For Short term 5-10 window size or 20-90 for Medium-Term Trading.
    Short-Term Trading:
        5-day or 10-day SMA
        Useful for swing traders or day traders
    Medium-Term Trading:
        20-day or 50-day SMA
        Useful for traders who hold positions for weeks or months
        Long-Term Trading:
        100-day or 200-day SMA
    
    Useful for long-term investors or positional traders
    """
    def __init__(self, data):
        self.data = data

    """
    Calculate the Simple Moving Average (SMA) based on the window size.
    :param window: The window size for the rolling average.
    :return: A Series of SMA values.
    """
    def calculate(self, window):
        # Calculate SMA
        return self.data['close'].rolling(window=window).mean()
    
    def calculate_trend_strength(self):
        """
        Calculate the trend strength based on short-term and long-term Simple Moving Averages (SMA).
        Parameters:
            short_term_sma (pd.Series): Short-term SMA values.
            long_term_sma (pd.Series): Long-term SMA values.
        Returns:
            pd.Series: Trend strength values.
        """
        # Calculate short-term SMA with a 20-day window
        short_term_sma = self.data['close'].rolling(window=20).mean()

        # Calculate long-term SMA with a 100-day window
        long_term_sma = self.data['close'].rolling(window=100).mean()

        # Calculate the trend strength
        trend_strength = ((short_term_sma - long_term_sma) / long_term_sma) * 100

        return trend_strength

class BollingerBands:
    """
    Bollinger Bands Indicator Class.
    Bollinger Bands are volatility bands placed above and below a moving average.
    Volatility is based on the standard deviation, which changes as volatility increases or decreases.

    :param data: Pandas DataFrame containing stock data, assumed to have a column named 'close'.
    :param sma: Instance of SMA class for calculating the Simple Moving Average.
    """
    def __init__(self, data, sma):
        """
        Initialize BollingerBands with stock data and an SMA object.
        
        :param data: Pandas DataFrame containing stock data.
        :param sma: Instance of SMA class.
        """
        self.data = data
        self.sma = sma  # Here, sma is an instance of the SMA class

    def calculate(self, window):
        """
        Calculate Bollinger Bands based on the window size.
        
        :param window: Window size for the rolling average and standard deviation.
        :return: A tuple containing the lower band, middle band (SMA), and upper band as Pandas Series.
        """
        sma_data = self.sma.calculate(window)
        rolling_std = self.data['close'].rolling(window=window).std()
        upper_band = sma_data + (rolling_std * 2)
        lower_band = sma_data - (rolling_std * 2)
        return lower_band, sma_data, upper_band

class SupportResistanceLevels:
    """
    Support and Resistance Levels Indicator Class.
    Support is a price level where a downtrend can be expected to pause.
    Resistance is a price level where a uptrend can be expected to pause.
    
    :param data: Pandas DataFrame containing stock data, expected to have 'low' and 'high' columns.
    """
    def __init__(self, data):
        """
        Initialize SupportResistanceLevels with stock data.
        
        :param data: Pandas DataFrame containing stock data.
        """
        self.data = data

    def calculate_support_resistance(self, window):
        """
        Calculate support and resistance levels.
        
        :param window: Window size for the rolling minimum and maximum.
        :return: A tuple containing the support and resistance levels as Pandas Series.
        """
        support = self.data['low'].rolling(window=window).min()
        resistance = self.data['high'].rolling(window=window).max()
        return support, resistance
    
class Volume:
    """
    Volume Indicator Class.
    Measures the number of shares or contracts traded in a security or market during a given period.
    
    :param data: Pandas DataFrame containing stock data, expected to have a 'volume' column.
    """
    def __init__(self, data):
        """
        Initialize Volume with stock data.
        
        :param data: Pandas DataFrame containing stock data.
        """
        self.data = data
    
    def average_volume(self, window):
        """
        Calculate average volume.
        
        :param window: Window size for the rolling average.
        :return: A Pandas Series containing average volume.
        """
        return self.data['volume'].rolling(window=window).mean()
    
class VWAP:
    """
    Volume Weighted Average Price (VWAP) Indicator Class.
    It gives an idea of the true average price a stock has traded at over a certain time frame.
    
    :param data: Pandas DataFrame containing stock data, expected to have 'close' and 'volume' columns.
    """
    def __init__(self, data):
        """
        Initialize VWAP with stock data.
        
        :param data: Pandas DataFrame containing stock data.
        """
        self.data = data
    
    def calculate_vmap(self, window):
        """
        Calculate VWAP.
        
        :param window: Window size for the rolling sum.
        :return: A Pandas Series containing VWAP values.
        """
        vwap = (self.data['close'] * self.data['volume']).rolling(window=window).sum() / self.data['volume'].rolling(window=window).sum()
        return vwap

class RSI:
    """
    Relative Strength Index (RSI) Indicator Class.
    Measures the speed and change of price movements.
    
    :param data: Pandas DataFrame containing stock data, expected to have a 'close' column.
    """
    def __init__(self, data):
        """
        Initialize RSI with stock data.
        
        :param data: Pandas DataFrame containing stock data.
        """
        self.data = data

    def calculate_rsi(self, window):
        """
        Calculate RSI.
        
        :param window: Window size for the rolling mean.
        :return: A Pandas Series containing RSI values.
        """
        delta = self.data['close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi




