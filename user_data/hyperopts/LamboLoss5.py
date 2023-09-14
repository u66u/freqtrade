"""
MaxDrawDownHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""
from datetime import datetime
import numpy as np
from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.data.metrics import calculate_expectancy, calculate_max_drawdown
from freqtrade.optimize.hyperopt import IHyperOptLoss
import math

# Set maximum expectancy used in the calculation
max_expectancy = 1
max_profit_ratio = 5
max_avg_profit = 200

class LamboLoss5(IHyperOptLoss):

    """
    Defines the loss function for hyperopt.
    
    This implementation optimizes for max drawdown, average profit, profit factor, expectancy, and high trade count
    Less max drawdown more profit -> Lower return value
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime, config: Config,
                               *args, **kwargs) -> float:

        """
        Objective function.

        Uses profit ratio weighted max_drawdown when drawdown is available.
        Otherwise directly optimizes profit ratio.
        """
        # total_profit = results['profit_abs'].sum()

        starting_balance = config['dry_run_wallet']
        stake_amount = config['stake_amount']
        max_profit_abs = (max_avg_profit / 100) * stake_amount

        strict_profit_abs = np.minimum(max_profit_abs, results['profit_abs'])
        results['profit_abs'] = strict_profit_abs

        total_profit = strict_profit_abs / starting_balance

        average_profit = total_profit.mean() * 100

        winning_profit = results.loc[results['profit_abs'] > 0, 'profit_abs'].sum()
        losing_profit = results.loc[results['profit_abs'] < 0, 'profit_abs'].sum()
        profit_factor = winning_profit / abs(losing_profit) if losing_profit else 10

        total_profit = strict_profit_abs.sum()

        expectancy, expectancy_ratio = calculate_expectancy(results)

        total_trades = len(results)

        trade_duration = results['trade_duration'].mean()
        if trade_duration == 0:
            trade_duration = 1
        # if (nb_loss_trades == 0):
        #     return -total_profit * 100
        
        loss_value = total_profit * min(average_profit, max_avg_profit) * min(profit_factor, max_profit_ratio) * min(expectancy_ratio, max_expectancy) * total_trades / math.sqrt(max(trade_duration, 5))

        if (total_profit < 0) and (loss_value > 0):
            return loss_value

        return (-1 * loss_value)
