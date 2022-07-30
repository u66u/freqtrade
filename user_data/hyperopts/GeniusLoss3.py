from freqtrade.optimize.hyperopt import IHyperOptLoss
import math
from datetime import datetime
from pandas import DataFrame, date_range
import pandas as pd
from freqtrade.data.metrics import calculate_max_drawdown, calculate_expectancy

# Sortino settings
TARGET_TRADES = 500
EXPECTED_MAX_PROFIT = 3.0 # x 100%
MAX_ACCEPTED_TRADE_DURATION = 180 # minutes
MIN_ACCEPTED_TRADE_DURATION = 2 # minutes
MAX_ACCEPTED_AVERAGE_TRADE_DAILY = 2

AVERAGE_PROFIT_THRESHOLD = 0.5 # 50%

IGNORE_SMALL_PROFITS = False
SMALL_PROFITS_THRESHOLD = 0.001  # 0.1%

class GeniusLoss3(IHyperOptLoss):
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
        profit_threshold = 0

        if IGNORE_SMALL_PROFITS:
            profit_threshold = SMALL_PROFITS_THRESHOLD

        # total_profit = results['profit_ratio'].sum()
        total_profit = results['profit_abs'].sum()
        total_trades = len(results)
        # total_win = len(results[(results['profit_ratio'] > profit_threshold)])
        # total_lose = len(results[(results['profit_ratio'] <= 0)])
        average_profit = results['profit_ratio'].mean()
        trade_duration = results['trade_duration'].mean()
        backtest_days = (max_date - min_date).days or 1
        average_trades_per_day = round(total_trades / backtest_days, 5)

        max_drawdown = 0
        try:
            max_drawdown = calculate_max_drawdown(results, value_col='profit_abs')[0]
        except:
            pass

        # if total_lose == 0:
        #     total_lose = 1

        profit_loss = total_profit
        average_profit_loss = (average_profit)
        drawdown_loss = max_drawdown if (max_drawdown > 0) else 1
        duration_loss = (trade_duration)
        average_trade_daily_loss = (average_trades_per_day)

        # result = profit_loss + win_lose_loss + average_profit_loss + sortino_ratio_loss + drawdown_loss + duration_loss
        expectancy = calculate_expectancy(results)

        result = (-profit_loss * average_profit_loss * average_trade_daily_loss) / (duration_loss * drawdown_loss * 10)

        if (expectancy < 0):
            if (result > 0):
                expectancy = expectancy * -1

        return (result * expectancy)
