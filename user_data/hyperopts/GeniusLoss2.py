from freqtrade.optimize.hyperopt import IHyperOptLoss
import math
from datetime import datetime
from pandas import DataFrame, date_range
import pandas as pd
from freqtrade.data.metrics import calculate_max_drawdown

# Sortino settings
TARGET_TRADES = 500
EXPECTED_MAX_PROFIT = 3.0 # x 100%
MAX_ACCEPTED_TRADE_DURATION = 180 # minutes
MIN_ACCEPTED_TRADE_DURATION = 2 # minutes
MIN_ACCEPTED_AVERAGE_TRADE_DAILY = 0.5
MIN_ACCEPTED_AVERAGE_PROFIT = 1.5

# Loss settings
# EXPECTED_MAX_PROFIT = 3.0
# WIN_LOSS_WEIGHT = 2
AVERAGE_PROFIT_WEIGHT = 1.5
AVERAGE_PROFIT_THRESHOLD = 20 # %
UNREALISTIC_AVERAGE_PROFIT = 50
SORTINO_WEIGHT = 0.2
TOTAL_PROFIT_WEIGHT = 1
DRAWDOWN_WEIGHT = 2
DURATION_WEIGHT = 1
AVERAGE_TRADE_DAILY_WEIGHT = 0.5

IGNORE_SMALL_PROFITS = False
SMALL_PROFITS_THRESHOLD = 0.001  # 0.1%

class GeniusLoss2(IHyperOptLoss):
    """
    Defines custom loss function which consider various metrics
    to make more robust strategy.
    Adjust those weights to get more suitable results for your strategy
    WIN_LOSS_WEIGHT
    AVERAGE_PROFIT_WEIGHT
    AVERAGE_PROFIT_THRESHOLD - upper threshold of average profit to rely on (cut off crazy av.profits like 10%+)
    SORTINO_WEIGHT
    TOTAL_PROFIT_WEIGHT


    IGNORE_SMALL_PROFITS - this param allow to filter small profits
    (to take into consideration possible spread)
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               *args, **kwargs) -> float:
        """
        Objective function, returns smaller number for better results.
        """

        total_profit = results['profit_abs'].sum()
        total_trades = len(results)
        average_profit = results['profit_ratio'].mean() * 100

        if average_profit > UNREALISTIC_AVERAGE_PROFIT:
            return 1000

        trade_duration = results['trade_duration'].mean()
        backtest_days = (max_date - min_date).days or 1
        average_trades_per_day = round(total_trades / backtest_days, 5)

        if (total_profit < 0) and (average_profit < 0):
            average_profit = -average_profit
            
        result = -total_profit * (average_profit/MIN_ACCEPTED_AVERAGE_PROFIT) * average_trades_per_day / trade_duration

        return result
