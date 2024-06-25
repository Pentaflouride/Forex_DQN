import MetaTrader5 as mt5
import numpy as np
import tulipy as ti
from datetime import datetime, timedelta

def initialize_mt5():
    """Initialize the MetaTrader5 connection."""
    if not mt5.initialize():
        print("MetaTrader5 initialization failed")
        return False
    return True

def get_current_price(symbol, lookback=1):
    """
    Get the current price(s) for a given symbol.

    Args:
    symbol (str): The trading symbol.
    lookback (int): Number of historical prices to return.

    Returns:
    list: List of current prices.
    """
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, lookback)
    if len(rates) < lookback:
        print(f"Insufficient data for {symbol}")
        return None
    return [rate['close'] for rate in rates][::-1]

def get_percentage_change(symbol, minutes, lookback=1):
    """
    Calculate the percentage change in price over given time periods.

    Args:
    symbol (str): The trading symbol.
    minutes (int): The time period for each percentage change calculation.
    lookback (int): Number of percentage changes to calculate.

    Returns:
    list: List of percentage changes.
    """
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, minutes * lookback + 1)

    if len(rates) < minutes * lookback + 1:
        print(f"Insufficient data for {symbol}")
        return None

    changes = []
    for i in range(lookback):
        start_price = rates[i * minutes]['close']
        end_price = rates[(i + 1) * minutes]['close']
        change = ((end_price - start_price) / start_price) * 100
        changes.append(change)
    return changes[::-1]

def get_rsi(symbol, period, lookback=1, timeframe=mt5.TIMEFRAME_M1):
    """
    Calculate the RSI for a given symbol.

    Args:
    symbol (str): The trading symbol.
    period (int): The RSI period.
    lookback (int): Number of RSI values to return.
    timeframe: The timeframe for data.

    Returns:
    list: List of RSI values.
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period + lookback)
    if len(rates) < period + lookback:
        print(f"Insufficient data for {symbol}")
        return None

    close_prices = np.array([rate['close'] for rate in rates])
    rsi = ti.rsi(close_prices, period=period)
    return list(rsi[-lookback:])[::-1]

def get_ema(symbol, period, lookback=1, timeframe=mt5.TIMEFRAME_M1):
    """
    Calculate the EMA for a given symbol.

    Args:
    symbol (str): The trading symbol.
    period (int): The EMA period.
    lookback (int): Number of EMA values to return.
    timeframe: The timeframe for data.

    Returns:
    list: List of EMA values.
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period + lookback - 1)
    if len(rates) < period + lookback - 1:
        print(f"Insufficient data for {symbol}")
        return None

    close_prices = np.array([rate['close'] for rate in rates])
    ema = ti.ema(close_prices, period=period)
    return list(ema[-lookback:])[::-1]

def get_stochastic(symbol, k_period, d_period, smooth_k, lookback=1, timeframe=mt5.TIMEFRAME_M1):
    """
    Calculate the Stochastic Oscillator for a given symbol.

    Args:
    symbol (str): The trading symbol.
    k_period (int): The %K period.
    d_period (int): The %D period.
    smooth_k (int): The %K smoothing period.
    lookback (int): Number of Stochastic values to return.
    timeframe: The timeframe for data.

    Returns:
    tuple: (List of %K values, List of %D values)
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, k_period + d_period + lookback - 1)
    if len(rates) < k_period + d_period + lookback - 1:
        print(f"Insufficient data for {symbol}")
        return None, None

    high_prices = np.array([rate['high'] for rate in rates])
    low_prices = np.array([rate['low'] for rate in rates])
    close_prices = np.array([rate['close'] for rate in rates])

    try:
        stoch_k, stoch_d = ti.stoch(high_prices, low_prices, close_prices, k_period, smooth_k, d_period)
        return list(stoch_k[-lookback:])[::-1], list(stoch_d[-lookback:])[::-1]
    except Exception as e:
        #print(f"Error calculating Stochastic for {symbol}: {e}")
        return None, None

def get_ema_crossover_signal(symbol, fast_period, slow_period, timeframe=mt5.TIMEFRAME_M1):
    ema_fast = get_ema(symbol, fast_period, lookback=2, timeframe=timeframe)
    ema_slow = get_ema(symbol, slow_period, lookback=2, timeframe=timeframe)

    if ema_fast is None or ema_slow is None or len(ema_fast) < 2 or len(ema_slow) < 2:
        return 0

    if ema_fast[-2] <= ema_slow[-2] and ema_fast[-1] > ema_slow[-1]:
        return 1  # Bullish crossover
    elif ema_fast[-2] >= ema_slow[-2] and ema_fast[-1] < ema_slow[-1]:
        return -1  # Bearish crossover
    else:
        return 0  # No crossover

def get_stochastic_crossover_signal(symbol, k_period, d_period, smooth_k, timeframe=mt5.TIMEFRAME_M1):
    stoch_k, stoch_d = get_stochastic(symbol, k_period, d_period, smooth_k, lookback=2, timeframe=timeframe)

    if stoch_k is None or stoch_d is None or len(stoch_k) < 2 or len(stoch_d) < 2:
        return 0

    if stoch_k[-2] <= stoch_d[-2] and stoch_k[-1] > stoch_d[-1]:
        return 1  # Bullish crossover
    elif stoch_k[-2] >= stoch_d[-2] and stoch_k[-1] < stoch_d[-1]:
        return -1  # Bearish crossover
    else:
        return 0  # No crossover

def down_sample(data, interval):
    """
    Down-sample data to a larger time interval.

    Args:
    data (list): List of input data.
    interval (int): New interval in minutes (5, 10, or 30).

    Returns:
    list: Down-sampled data.
    """
    if interval not in [5, 10, 30]:
        print("Invalid interval. Use 5, 10, or 30.")
        return None

    samples = len(data) // interval
    return [data[i * interval] for i in range(samples)]

def shutdown_mt5():
    """Shut down the MetaTrader5 connection."""
    mt5.shutdown()
