import logging
import shutil
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pandas import DataFrame

from freqtrade.configuration import TimeRange
from freqtrade.data.history import get_timerange
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.loggers.set_log_levels import (reduce_verbosity_for_bias_tester,
                                              restore_verbosity_for_bias_tester)
from freqtrade.optimize.backtesting import Backtesting


logger = logging.getLogger(__name__)


class VarHolder:
    timerange: TimeRange
    data: DataFrame
    indicators: Dict[str, DataFrame]
    result: DataFrame
    compared: DataFrame
    from_dt: datetime
    to_dt: datetime
    compared_dt: datetime
    timeframe: str


class Analysis:
    def __init__(self) -> None:
        self.total_signals = 0
        self.false_entry_signals = 0
        self.false_exit_signals = 0
        self.false_indicators: List[str] = []
        self.has_bias = False


class RecursiveAnalysis:

    def __init__(self, config: Dict[str, Any], strategy_obj: Dict):
        self.failed_bias_check = True
        self.full_varHolder = VarHolder()
        self.partial_varHolder_array = []

        self.entry_varHolders: List[VarHolder] = []
        self.exit_varHolders: List[VarHolder] = []
        self.exchange: Optional[Any] = None

        # pull variables the scope of the recursive_analysis-instance
        self.local_config = deepcopy(config)
        self.local_config['strategy'] = strategy_obj['name']
        self.current_analysis = Analysis()
        self.minimum_trade_amount = config['minimum_trade_amount']
        self.targeted_trade_amount = config['targeted_trade_amount']
        self.strategy_obj = strategy_obj

    @staticmethod
    def dt_to_timestamp(dt: datetime):
        timestamp = int(dt.replace(tzinfo=timezone.utc).timestamp())
        return timestamp

    @staticmethod
    def get_result(backtesting: Backtesting, processed: DataFrame):
        min_date, max_date = get_timerange(processed)

        result = backtesting.backtest(
            processed=deepcopy(processed),
            start_date=min_date,
            end_date=max_date
        )
        return result

    @staticmethod
    def report_signal(result: dict, column_name: str, checked_timestamp: datetime):
        df = result['results']
        row_count = df[column_name].shape[0]

        if row_count == 0:
            return False
        else:

            df_cut = df[(df[column_name] == checked_timestamp)]
            if df_cut[column_name].shape[0] == 0:
                return False
            else:
                return True
        return False

    # analyzes two data frames with processed indicators and shows differences between them.
    def analyze_indicators(self, full_vars: VarHolder, cut_vars: VarHolder, current_pair: str):
        # extract dataframes
        cut_df: DataFrame = cut_vars.indicators[current_pair]
        full_df: DataFrame = full_vars.indicators[current_pair]

        # cut longer dataframe to length of the shorter
        full_df_cut = full_df[
            (full_df.date == cut_vars.compared_dt)
        ].reset_index(drop=True)
        cut_df_cut = cut_df[
            (cut_df.date == cut_vars.compared_dt)
        ].reset_index(drop=True)

        # check if dataframes are not empty
        if full_df_cut.shape[0] != 0 and cut_df_cut.shape[0] != 0:

            # compare dataframes
            compare_df = full_df_cut.compare(cut_df_cut)

            if compare_df.shape[0] > 0:
                for col_name, values in compare_df.items():
                    col_idx = compare_df.columns.get_loc(col_name)
                    compare_df_row = compare_df.iloc[0]
                    # compare_df now comprises tuples with [1] having either 'self' or 'other'
                    if 'other' in col_name[1]:
                        continue
                    self_value = compare_df_row[col_idx]
                    other_value = compare_df_row[col_idx + 1]

                    # output differences
                    if self_value != other_value:

                        if not self.current_analysis.false_indicators.__contains__(col_name[0]):
                            self.current_analysis.false_indicators.append(col_name[0])
                            logger.info(f"=> found look ahead bias in indicator "
                                        f"{col_name[0]}. "
                                        f"{str(self_value)} != {str(other_value)}")

    def prepare_data(self, varholder: VarHolder, pairs_to_load: List[DataFrame]):

        if 'freqai' in self.local_config and 'identifier' in self.local_config['freqai']:
            # purge previous data if the freqai model is defined
            # (to be sure nothing is carried over from older backtests)
            path_to_current_identifier = (
                Path(f"{self.local_config['user_data_dir']}/models/"
                     f"{self.local_config['freqai']['identifier']}").resolve())
            # remove folder and its contents
            if Path.exists(path_to_current_identifier):
                shutil.rmtree(path_to_current_identifier)

        prepare_data_config = deepcopy(self.local_config)
        prepare_data_config['timerange'] = (str(self.dt_to_timestamp(varholder.from_dt)) + "-" +
                                            str(self.dt_to_timestamp(varholder.to_dt)))
        prepare_data_config['exchange']['pair_whitelist'] = pairs_to_load

        backtesting = Backtesting(prepare_data_config, self.exchange)
        self.exchange = backtesting.exchange
        backtesting._set_strategy(backtesting.strategylist[0])

        varholder.data, varholder.timerange = backtesting.load_bt_data()
        backtesting.load_bt_data_detail()
        varholder.timeframe = backtesting.timeframe

        varholder.indicators = backtesting.strategy.advise_all_indicators(varholder.data)
        varholder.result = self.get_result(backtesting, varholder.indicators)

    def fill_full_varholder(self):
        self.full_varHolder = VarHolder()

        # define datetime in human-readable format
        parsed_timerange = TimeRange.parse_timerange(self.local_config['timerange'])

        if parsed_timerange.startdt is None:
            self.full_varHolder.from_dt = datetime.fromtimestamp(0, tz=timezone.utc)
        else:
            self.full_varHolder.from_dt = parsed_timerange.startdt

        if parsed_timerange.stopdt is None:
            self.full_varHolder.to_dt = datetime.utcnow()
        else:
            self.full_varHolder.to_dt = parsed_timerange.stopdt

        self.prepare_data(self.full_varHolder, self.local_config['pairs'])

    def fill_partial_varholder(self, start_date):
        partial_varHolder = VarHolder()

        partial_varHolder.from_dt = start_date
        partial_varHolder.to_dt = self.full_varHolder.to_dt

        self.prepare_data(partial_varHolder, self.local_config['pairs'])

        self.partial_varHolder_array.append(partial_varHolder)

    def start(self) -> None:

        # first make a single backtest
        self.fill_full_varholder()

        reduce_verbosity_for_bias_tester()

        start_date_full = self.full_varHolder.from_dt
        end_date_full = self.full_varHolder.to_dt

        timeframe_minutes = timeframe_to_minutes(self.full_varHolder.timeframe)

        start_date_partial = []
        add_date_partial = True
        i = 125

        while add_date_partial:
            temp_date = end_date_full - timedelta(minutes=int(timeframe_minutes * i))
            if(temp_date.timestamp() > start_date_full.timestamp()):
                start_date_partial.append(temp_date)
                i = i * 2
            else:
                break

        for start_date in start_date_partial:
            self.fill_partial_varholder(start_date)


        pair_to_check = self.local_config['pairs'][0]

        # for idx, result_row in self.full_varHolder.result['results'].iterrows():
        #     pair_to_check = result_row['pair']
        #     break

        # Restore verbosity, so it's not too quiet for the next strategy
        restore_verbosity_for_bias_tester()
        logger.info(f"Start checking for recursive bias")
        # check and report signals
        base_last_row = self.full_varHolder.indicators[pair_to_check].iloc[-1]
        base_timerange = self.full_varHolder.from_dt.strftime('%Y-%m-%dT%H:%M:%S') + "-" + self.full_varHolder.to_dt.strftime('%Y-%m-%dT%H:%M:%S')
        for part in self.partial_varHolder_array:
            part_last_row = part.indicators[pair_to_check].iloc[-1]
            part_timerange = part.from_dt.strftime('%Y-%m-%dT%H:%M:%S') + "-" + part.to_dt.strftime('%Y-%m-%dT%H:%M:%S')

            logger.info(f"Comparing last row of {base_timerange} vs {part_timerange}")
            compare_df = base_last_row.compare(part_last_row)
            if compare_df.shape[0] > 0:
                print(compare_df)
                # for col_name, values in compare_df.items():
                    # logger.info(f"0 {col_name[0]}, 1 {col_name[1]}")
                    # col_idx = compare_df.columns.get_loc(col_name)
                    # compare_df_row = compare_df.iloc[0]
                    # # compare_df now comprises tuples with [1] having either 'self' or 'other'
                    # if 'other' in col_name[1]:
                    #     continue
                    # self_value = compare_df_row[col_idx]
                    # other_value = compare_df_row[col_idx + 1]

                    # # output differences
                    # if self_value != other_value:
                    #     difference = (other_value - self_value) / self_value * 100
                    #     logger.info(f"=> found difference in indicator "
                    #                 f"{col_name[0]}, with difference of "
                    #                 "{:.8f}%".format(difference))
            else:
                logger.info("No difference found. Moving to next comparison")

        # if self.current_analysis.total_signals < self.local_config['minimum_trade_amount']:
        #     logger.info(f" -> {self.local_config['strategy']} : too few trades. "
        #                 f"We only found {self.current_analysis.total_signals} trades. "
        #                 f"Hint: Extend the timerange "
        #                 f"to get at least {self.local_config['minimum_trade_amount']} "
        #                 f"or lower the value of minimum_trade_amount.")
        #     self.failed_bias_check = True
        # elif (self.current_analysis.false_entry_signals > 0 or
        #       self.current_analysis.false_exit_signals > 0 or
        #       len(self.current_analysis.false_indicators) > 0):
        #     logger.info(f" => {self.local_config['strategy']} : bias detected!")
        #     self.current_analysis.has_bias = True
        #     self.failed_bias_check = False
        # else:
        #     logger.info(self.local_config['strategy'] + ": no bias detected")
        #     self.failed_bias_check = False
