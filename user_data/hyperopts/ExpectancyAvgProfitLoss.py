"""
MaxDrawDownHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""
from datetime import datetime

from pandas import DataFrame

from freqtrade.data.metrics import calculate_expectancy
from freqtrade.optimize.hyperopt import IHyperOptLoss


class ExpectancyAvgProfitLoss(IHyperOptLoss):

    """
    Defines the loss function for hyperopt.

    This implementation optimizes for max draw down and profit
    Less max drawdown more profit -> Lower return value
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               *args, **kwargs) -> float:

        """
        Objective function.

        Uses profit ratio weighted max_drawdown when drawdown is available.
        Otherwise directly optimizes profit ratio.
        """
        # total_profit = results['profit_abs'].sum()

        average_profit = results['profit_ratio'].mean() * 100

        expectancy = calculate_expectancy(results)

        nb_loss_trades = len(results.loc[results['profit_abs'] < 0])

        if (nb_loss_trades == 0):
            return 0

        if (average_profit < 0) and (expectancy < 0):
            expectancy = expectancy * -1
            
        return  (-1 * average_profit * expectancy)
