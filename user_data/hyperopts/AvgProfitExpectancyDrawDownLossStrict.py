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

# Set maximum expectancy used in the calculation
max_expectancy = 2
max_avg_profit = 15

class AvgProfitExpectancyDrawDownLossStrict(IHyperOptLoss):

    """
    Defines the loss function for hyperopt.

    This implementation optimizes for max draw down and profit
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
        max_profit_abs = 0.15 * stake_amount

        strict_profit_abs = np.minimum(max_profit_abs, results['profit_abs'])

        total_profit = strict_profit_abs / starting_balance

        average_profit = total_profit.mean() * 100

        total_profit = strict_profit_abs.sum()

        expectancy = calculate_expectancy(results)

        try:
            max_drawdown = calculate_max_drawdown(results, value_col='profit_abs')
        except ValueError:
            # No losing trade, therefore no drawdown.
            return -total_profit * 100
        
        loss_value = total_profit * min(average_profit, max_avg_profit) * min(expectancy, max_expectancy) / max_drawdown[0]

        if (total_profit < 0) and (loss_value > 0):
            return loss_value

        return (-1 * loss_value)
