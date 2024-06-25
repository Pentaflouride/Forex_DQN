import numpy as np
from forex_indicators import get_current_price, get_percentage_change, get_rsi, get_ema, get_stochastic, get_ema_crossover_signal, get_stochastic_crossover_signal
import MetaTrader5 as mt5
from datetime import datetime,timedelta
import os
import pickle


class ForexEnv:
    """
    A Forex trading environment for reinforcement learning.

    This environment simulates a day of Forex trading, allowing an agent to make
    buy, sell, or hold decisions based on various market indicators.

    Attributes:
        symbol (str): The Forex symbol being traded (e.g., "EURUSD").
        initial_balance (float): The starting balance for each episode.
        leverage (float): The trading leverage used.
        commission (float): The commission rate for each trade.
        day_data (list): List of market data for the current trading day.
        current_step (int): The current step within the day's data.
    """

    def __init__(self, symbol, initial_balance=10000, leverage=100, commission=0.0001, episode_length=1440):
        """
        Initialize the Forex trading environment.

        Args:
            symbol (str): The Forex symbol to trade.
            initial_balance (float): Starting balance for each episode.
            leverage (float): Trading leverage.
            commission (float): Commission rate for trades.
            episode_length (int): Length of each episode in minutes.
        """
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.commission = commission
        self.position = 0  # 0: no position, 1: long position
        self.entry_price = 0
        self.day_data = []
        self.current_step = 0
        self.episode_length = episode_length  # New attribute
        self.trading_hours = range(9, 17)
        self.current_episode_num = 0
        self.current_date = self._get_last_trading_day()

        if not mt5.initialize():
            print("MetaTrader5 initialization failed")
            mt5.shutdown()
            raise Exception("MT5 initialization failed")

    def __del__(self):
        mt5.shutdown()

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.current_step = 0
        self.day_data = self._get_new_hour_data(self.symbol, self.current_episode_num)
        if self.day_data is None:
            raise Exception("Failed to fetch day data")
        self.current_episode_num += 1  # Move to the next episode
        return self._get_observation()

    def _get_last_trading_day(self):
        current_date = datetime.now().date()
        while current_date.weekday() > 4:  # Find the most recent weekday (Monday to Friday)
            current_date -= timedelta(days=1)
        return current_date

    def _get_new_hour_data(self, symbol, episode_num):
        # Calculate the hour for the current episode
        start_hour_index = episode_num % len(self.trading_hours)
        start_hour = self.trading_hours[start_hour_index]

        # Move to the previous day if all hours of the current day have been used
        if start_hour_index == 0 and episode_num > 0:
            self.current_date -= timedelta(days=1)
            while self.current_date.weekday() > 4:  # Skip weekends
                self.current_date -= timedelta(days=1)

        start_time = datetime.combine(self.current_date, datetime.min.time()) + timedelta(hours=start_hour)
        end_time = start_time + timedelta(hours=1)
        print(start_time, end_time)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start_time, end_time)

        if rates is None or len(rates) == 0:
            print(f"Failed to fetch data for {symbol} on {self.current_date}")
            return None

        processed_data = []
        for i, rate in enumerate(rates):
            close_price = rate['close']
            pct_change = 0 if i == 0 else (close_price - rates[i - 1]['close']) / rates[i - 1]['close'] * 100
            rsi = get_rsi(symbol, period=14, timeframe=mt5.TIMEFRAME_M1)[-1]
            ema_fast = get_ema(symbol, period=50, timeframe=mt5.TIMEFRAME_M1)[-1]
            ema_slow = get_ema(symbol, period=100, timeframe=mt5.TIMEFRAME_M1)[-1]
            stoch_k, stoch_d = get_stochastic(symbol, k_period=14, d_period=3, smooth_k=3, timeframe=mt5.TIMEFRAME_M1)
            ema_cross = get_ema_crossover_signal(symbol, fast_period=50, slow_period=100, timeframe=mt5.TIMEFRAME_M1)
            stoch_cross = get_stochastic_crossover_signal(symbol, k_period=14, d_period=3, smooth_k=3, timeframe=mt5.TIMEFRAME_M1)

            processed_data.append({
                'close': close_price,
                'pct_change': pct_change,
                'rsi': rsi,
                'ema_fast': ema_fast,
                'ema_slow': ema_slow,
                'stoch_k': stoch_k[-1] if stoch_k is not None else None,
                'stoch_d': stoch_d[-1] if stoch_d is not None else None,
                'ema_cross': ema_cross,
                'stoch_cross': stoch_cross
            })

        return processed_data

    def step(self, action):
        """
        Take a step in the environment based on the chosen action.

        Args:
            action (int): The action to take (0: Hold, 1: Buy, 2: Sell, 3: Wait).

        Returns:
            tuple: (observation, reward, done, info)
        """
        self.current_step += 1

        current_price = self.day_data[self.current_step]['close']

        reward = 0
        done = False

        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = current_price
            self.balance -= self.balance * self.commission  # Apply commission
        elif action == 2 and self.position == 1:  # Sell
            reward = (current_price - self.entry_price) / self.entry_price * self.leverage
            self.balance *= (1 + reward)
            self.balance -= self.balance * self.commission  # Apply commission
            self.position = 0
            self.entry_price = 0

        # Calculate unrealized profit/loss if in a position
        if self.position == 1:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price * self.leverage
            reward = unrealized_pnl

        # Check if episode should end
        if self.current_step >= len(self.day_data) - 1:
            done = True

        info = {
            'balance': self.balance,
            'position': self.position,
            'entry_price': self.entry_price,
            'current_price': current_price
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        """
        Get the current market observation.

        Returns:
            numpy.array: The current market state observation.
        """
        data = self.day_data[self.current_step]
        return np.array([
            data['close'],
            data['pct_change'],
            data['rsi'],
            data['ema_fast'],
            data['ema_slow'],
            data['stoch_k'],
            data['stoch_d'],
            data['ema_cross'],
            data['stoch_cross'],
            self.position,
            self.balance / self.initial_balance
        ])

    def _get_new_day_data(self, symbol):
        """
        Fetch a new day's worth of market data for the given symbol using MT5 API.

        Args:
            symbol (str): The Forex symbol to fetch data for.

        Returns:
            list: A list of dictionaries containing market data for each minute of the trading day.
        """
        # Get the current date
        current_date = datetime.now().date()

        # Find the most recent weekday (Monday to Friday)
        while current_date.weekday() > 4:  # 5 = Saturday, 6 = Sunday
            current_date -= timedelta(days=1)

        # Set the start and end time for the episode length
        start_time = datetime.combine(current_date, datetime.min.time())
        end_time = start_time + timedelta(minutes=self.episode_length)

        # Fetch minute data from MT5
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start_time, end_time)

        if rates is None or len(rates) == 0:
            print(f"Failed to fetch data for {symbol} on {current_date}")
            return None

        # Process the raw data and calculate indicators
        processed_data = []
        for i, rate in enumerate(rates):
            close_price = rate['close']

            # Calculate percentage change
            if i == 0:
                pct_change = 0
            else:
                pct_change = (close_price - rates[i - 1]['close']) / rates[i - 1]['close'] * 100

            # Calculate other indicators
            rsi = get_rsi(symbol, period=14, timeframe=mt5.TIMEFRAME_M1)[-1]
            ema_fast = get_ema(symbol, period=50, timeframe=mt5.TIMEFRAME_M1)[-1]
            ema_slow = get_ema(symbol, period=100, timeframe=mt5.TIMEFRAME_M1)[-1]
            stoch_k, stoch_d = get_stochastic(symbol, k_period=14, d_period=3, smooth_k=3, timeframe=mt5.TIMEFRAME_M1)
            ema_cross = get_ema_crossover_signal(symbol, fast_period=50, slow_period=100, timeframe=mt5.TIMEFRAME_M1)
            stoch_cross = get_stochastic_crossover_signal(symbol, k_period=14, d_period=3, smooth_k=3,
                                                          timeframe=mt5.TIMEFRAME_M1)

            processed_data.append({
                'close': close_price,
                'pct_change': pct_change,
                'rsi': rsi,
                'ema_fast': ema_fast,
                'ema_slow': ema_slow,
                'stoch_k': stoch_k[-1] if stoch_k is not None else None,
                'stoch_d': stoch_d[-1] if stoch_d is not None else None,
                'ema_cross': ema_cross,
                'stoch_cross': stoch_cross
            })

        return processed_data

    def render(self):
        """
        Render the current state of the environment.
        """
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position}")
