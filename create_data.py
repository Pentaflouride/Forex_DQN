import os
import pickle
from datetime import datetime, timedelta
import MetaTrader5 as mt5
popular_symbols = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
                   "USDCHF", "NZDUSD", "EURJPY", "EURGBP", "EURCHF",
                   "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "EURAUD",
                   "EURNZD", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD"]
import os
import pickle
from datetime import datetime, timedelta
import MetaTrader5 as mt5

def save_daily_data_for_symbols(symbols, days_back=20, folder="forex_data"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back - 1)

    current_date = start_date

    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() > 4:
            current_date += timedelta(days=1)
            continue

        for symbol in symbols:
            start_time = datetime.combine(current_date, datetime.min.time())
            end_time = start_time + timedelta(hours=23, minutes=59)

            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start_time, end_time)

            if rates is not None and len(rates) > 0:
                file_path = os.path.join(folder, f"{symbol}_{current_date}.pkl")
                with open(file_path, "wb") as f:
                    pickle.dump(rates, f)
                print(f"Saved data for {symbol} on {current_date}")
            else:
                print(f"Failed to fetch data for {symbol} on {current_date}")

        current_date += timedelta(days=1)

# Example usage
if mt5.initialize():
    save_daily_data_for_symbols(popular_symbols, days_back=20)
    mt5.shutdown()
else:
    print("MetaTrader5 initialization failed")