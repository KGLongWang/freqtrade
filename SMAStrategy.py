# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple
import logging
import pytz

import json

logger = logging.getLogger(__name__)

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes, 
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib

class SMAStrategy(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    position_adjustment_enable = True
    base_stake_amount = DecimalParameter(
        100, 1000, default=1000,
        space="buy", optimize=False,
        load=False  # 允许从配置文件加载
    )

    max_leverage = DecimalParameter(
        1, 32, default=32,
        space="buy", optimize=True,
        load=False
    )

    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "1m"

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {}

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.99

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 45

    # Strategy parameters
    # buy_rsi = IntParameter(10, 40, default=30, space="buy")
    # sell_rsi = IntParameter(60, 90, default=70, space="sell")# Optional order type mapping.
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False
    }

    # Optional order time in force.
    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC"
    }

    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            "main_plot": {
                "tema": {},
                "sar": {"color": "white"},
            },
            "subplots": {
                # Subplots - each dict defines one additional plot
                "MACD": {
                    "macd": {"color": "blue"},
                    "macdsignal": {"color": "orange"},
                },
                "RSI": {
                    "rsi": {"color": "red"},
                }
            }
        }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        pd.set_option('display.precision', 6)
        pd.set_option('display.float_format', lambda x: '%.6f' % x)

        self.notification_states = {}
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        self.sma_pairs = [("5", "10"), ("10", "30"), ("5", "30")]
        self.sma_periods = [5, 10, 30]
        self.trend_summary = {}
        for tf in self.timeframes:
            for period1, period2 in self.sma_pairs:
                cross_key = f"sma{period1}_{tf}_cross_sma{period2}_{tf}"
                self.notification_states[f"{cross_key}_last_signal"] = None

            there_cross_key = f"sma5_{tf}_cross_sma10_{tf}_cross_sma30_{tf}"
            self.notification_states[f"{there_cross_key}_last_signal"] = None
        logger.info(f"🔔 notification_states: {self.notification_states}")

    def bot_start(self, **kwargs) -> None:
        self.dp.send_msg(f"🤖 交易机器人启动成功！启动时间: {datetime.now()}")

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return [("DOGE/USDT:USDT", "1m"),
                ("DOGE/USDT:USDT", "5m"),
                ("DOGE/USDT:USDT", "15m"),
                ("DOGE/USDT:USDT", "1h"),
                ("DOGE/USDT:USDT", "4h"),
                ("DOGE/USDT:USDT", "1d"),
                ("DOGE/USDT:USDT", "1w"),
                ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        交易法主要指标
        """
        def get_h1_and_l1_cross(dataframe):
            dataframe['hh'] = (
                dataframe['high'] +
                dataframe['high'].rolling(2).max() * 2 +
                dataframe['high'].rolling(4).max() * 2 +
                dataframe['high'].rolling(8).max() * 2 +
                dataframe['high'].rolling(13).max() +
                dataframe['high'].rolling(40).max() +
                dataframe['high'].rolling(20).max()
            ) / 10

            dataframe['ll'] = (
                dataframe['low'] +
                dataframe['low'].rolling(2).min() * 2 +
                dataframe['low'].rolling(4).min() * 2 +
                dataframe['low'].rolling(8).min() * 2 +
                dataframe['low'].rolling(13).min() +
                dataframe['low'].rolling(40).min() +
                dataframe['low'].rolling(20).min()
            ) / 10

            h1_conditions = (
                (dataframe['hh'] < dataframe['hh'].shift(1)) &
                (dataframe['ll'] < dataframe['ll'].shift(1)) &
                (dataframe['open'].shift(1) > dataframe['close']) &
                (dataframe['open'] > dataframe['close'])
            )
            l1_conditions = (
                (dataframe['hh'] > dataframe['hh'].shift(1)) &
                (dataframe['ll'] > dataframe['ll'].shift(1)) &
                (dataframe['open'].shift(1) < dataframe['close']) &
                (dataframe['open'] < dataframe['close'])
            )

            dataframe['h1'] = np.where(
                h1_conditions,
                dataframe['hh'].shift(2),
                np.nan
            )
            dataframe['l1'] = np.where(
                l1_conditions,
                dataframe['ll'].shift(2),
                np.nan
            )

            dataframe['h1'] = dataframe['h1'].ffill()
            dataframe['l1'] = dataframe['l1'].ffill()

            dataframe['h1_or_l1_cross'] = np.where(
                qtpylib.crossed(dataframe['close'], dataframe['h1'], direction='above'),
                1,
                np.where(qtpylib.crossed(dataframe['close'], dataframe['l1'], direction='below'), 0, np.nan)
            )
            dataframe['h1_or_l1_cross_state'] = dataframe['h1_or_l1_cross']
            dataframe['h1_or_l1_cross_state'] = dataframe['h1_or_l1_cross_state'].ffill() #1 是上穿， 0 是下穿

            return dataframe

        dataframe = get_h1_and_l1_cross(dataframe)

        inf_1m = self.dp.get_pair_dataframe(metadata['pair'], '1m')
        inf_5m = self.dp.get_pair_dataframe(metadata['pair'], '5m')
        inf_15m = self.dp.get_pair_dataframe(metadata['pair'], '15m')
        inf_1h = self.dp.get_pair_dataframe(metadata['pair'], '1h')
        inf_4h = self.dp.get_pair_dataframe(metadata['pair'], '4h')
        inf_1d = self.dp.get_pair_dataframe(metadata['pair'], '1d')
        inf_1w = self.dp.get_pair_dataframe(metadata['pair'], '1w')

        inf_1m = inf_1m.add_suffix('_1m')
        inf_5m = inf_5m.add_suffix('_5m')
        inf_15m = inf_15m.add_suffix('_15m')
        inf_1h = inf_1h.add_suffix('_1h')
        inf_4h = inf_4h.add_suffix('_4h')
        inf_1d = inf_1d.add_suffix('_1d')
        inf_1w = inf_1w.add_suffix('_1w')

        target_index = inf_1m.index
        inf_1d_new_index = target_index[-len(inf_1d):]
        inf_1d = inf_1d.set_index(inf_1d_new_index)
        inf_1d = inf_1d.reindex(target_index)

        inf_1w_new_index = target_index[-len(inf_1w):]
        inf_1w = inf_1w.set_index(inf_1w_new_index)
        inf_1w = inf_1w.reindex(target_index)

        merged_pair = pd.concat([dataframe, inf_1m, inf_5m, inf_15m, inf_1h, inf_4h, inf_1d, inf_1w], axis=1)

        try:
            ticker = self.dp.ticker(metadata['pair'])
        except Exception as e:
            self.dp.send_msg(f"🔔 获取ticker数据失败: {e}", always_send=True)
            logger.error(f"获取ticker数据失败: {e}")
            return dataframe
        
        #MA计算
        current_close = ticker['last']
        for tf in self.timeframes:
            rt_close = (pd.concat([merged_pair[f'close_{tf}'], pd.Series(current_close)]).iloc[1:])
            for period in self.sma_periods:
                sma_name = f'sma{period}_{tf}'   #column_name = 'sma5_1m'
                merged_pair[sma_name] = ta.SMA(rt_close, timeperiod=period).round(6)
                logger.info(f"sma_name: {sma_name}, sma_value: {merged_pair[sma_name]}")

        for tf in self.timeframes:
            for period1, period2 in self.sma_pairs:
                cross_dataframe = detect_ma_crossover(merged_pair,
                    f'sma{period1}_{tf}',
                    f'sma{period2}_{tf}'
                )
            cross_dataframe = detect_triple_ma_cross(cross_dataframe, f'sma5_{tf}', f'sma10_{tf}', f'sma30_{tf}')
            self.update_trend_summary(cross_dataframe, tf)
        self.trend_msg = get_trend_msg(self.trend_summary)

        for tf in self.timeframes:
            self.check_cross_alert(cross_dataframe, tf, current_close)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 1周均线测试
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 1周均线测试
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        return dataframe

    def get_cross_msg(self, tf: str, period1: str, period2: str, current_close, is_golden: bool):
        """发送单个均线交叉提醒"""
        msg = f"""
⚡️ 均线穿越提醒 
━━━━━━━━━━━━━━━
📊 时间周期: {tf}
⚡️ 信号类型: {'金叉' if is_golden else '死叉'}
🎯 具体表现: MA{period1}均线{'上穿' if is_golden else '下穿'}MA{period2}均线
💹 当前价格: {current_close}
━━━━━━━━━━━━━━━
"""
        return msg

    def get_three_cross_msg(self, tf, current_close, is_golden: bool):
        msg = f"""
🔔🔔 三线穿越提醒 
━━━━━━━━━━━━━━━
📊 时间周期: {tf}
⚡️ 信号类型: {'金叉' if is_golden else '死叉'}
🎯 具体表现:  {'多头排列' if is_golden else '空头排列'}
💹 当前价格: {current_close}
━━━━━━━━━━━━━━━
"""
        return msg
    
    def update_trend_summary(self, dataframe, tf):
        latest = dataframe.iloc[-1]
        if latest[f'sma5_{tf}_cross_sma10_{tf}_cross_sma30_{tf}_state'].item() == 1:
            trend = "多头排列"
        elif latest[f'sma5_{tf}_cross_sma10_{tf}_cross_sma30_{tf}_state'].item() == 0:
            trend = "空头排列" 
        else:
            trend = "未形成排列"
        self.trend_summary[tf] = trend

    def check_cross_alert(self, dataframe, tf, current_close):
        latest = dataframe.iloc[-1]
        for period1, period2 in self.sma_pairs:
            cross_key = f"sma{period1}_{tf}_cross_sma{period2}_{tf}"
                # 检查金叉
            if latest[cross_key].item()  == 1 and self.notification_states[f"{cross_key}_last_signal"] != 1:
                alert_msg = self.trend_msg + self.get_cross_msg(tf, period1, period2, current_close, True)
                self.dp.send_msg(alert_msg)
                self.notification_states[f"{cross_key}_last_signal"] = 1

                # 检查死叉
            elif latest[cross_key].item() == 0 and self.notification_states[f"{cross_key}_last_signal"] != 0: 
                cross_msg = self.get_cross_msg(tf, period1, period2, current_close, False) 
                alert_msg = self.trend_msg + cross_msg
                self.dp.send_msg(alert_msg)
                self.notification_states[f"{cross_key}_last_signal"] = 0

        there_cross_key = f"sma5_{tf}_cross_sma10_{tf}_cross_sma30_{tf}"
        if latest[there_cross_key].item() == 1 and self.notification_states[f"{there_cross_key}_last_signal"] != 1:
            there_cross_msg = self.get_three_cross_msg(tf, current_close, True)
            alert_msg = self.trend_msg + there_cross_msg
            self.dp.send_msg(alert_msg)
            self.notification_states[f"{there_cross_key}_last_signal"] = 1

        elif latest[there_cross_key].item() == 0 and self.notification_states[f"{there_cross_key}_last_signal"] != 0:
            there_cross_msg = self.get_three_cross_msg(tf, current_close, False)
            alert_msg = self.trend_msg + there_cross_msg
            self.dp.send_msg(alert_msg)
            self.notification_states[f"{there_cross_key}_last_signal"] = 0 

def get_trend_msg(trend_summary):
    msg = "━━━━━━━━━━━━━━━\n"
    msg += "🔔 各周期均线排列状态\n"
    msg += "━━━━━━━━━━━━━━━\n"
    msg += f"⏰ 1MIN: {trend_summary.get('1m', 'NA')}\n"
    msg += f"⏰ 5MIN: {trend_summary.get('5m', 'NA')}\n"
    msg += f"⏰ 15MIN: {trend_summary.get('15m', 'NA')}\n"
    msg += f"⏰ 1H: {trend_summary.get('1h', 'NA')}\n"
    msg += f"⏰ 4H: {trend_summary.get('4h', 'NA')}\n"
    msg += f"⏰ 1D: {trend_summary.get('1d', 'NA')}\n"
    msg += "━━━━━━━━━━━━━━━\n"

    return msg

def detect_ma_crossover(dataframe: pd.DataFrame, fast_ma: str, slow_ma: str) -> pd.DataFrame:
    """
    检测移动平均线交叉
    
    参数:
        dataframe: 数据框
        fast_ma: 快速移动平均线列名
        slow_ma: 慢速移动平均线列名
    
    返回:
        添加了交叉信号列的数据框
    """
    column_name = f'{fast_ma}_cross_{slow_ma}'
    cross_state = f'{fast_ma}_cross_{slow_ma}_state'
    # 检测上穿和下穿
    cross_up = qtpylib.crossed(dataframe[fast_ma], dataframe[slow_ma], direction='above')
    cross_down = qtpylib.crossed(dataframe[fast_ma], dataframe[slow_ma], direction='below')
    
    # 初始化信号列并填充交叉信号
    dataframe[column_name] = np.nan
    dataframe.loc[cross_up, column_name] = 1
    dataframe.loc[cross_down, column_name] = 0
    # 生成并维护状态
    dataframe[cross_state] = dataframe[column_name]
    dataframe[cross_state] = dataframe[cross_state].ffill()
    return dataframe

def detect_triple_ma_cross(dataframe: pd.DataFrame, ma1: str, ma2: str, ma3: str) -> pd.DataFrame:
    """
    检测三条移动平均线的交叉关系和状态
    
    参数:
        dataframe: 数据框
        ma1: 第一条均线(最快)
        ma2: 第二条均线
        ma3: 第三条均线(最慢)
    
    返回:
        添加了三线交叉信号和状态的数据框
    """
    # 定义列名
    cross_name = f'{ma1}_cross_{ma2}_cross_{ma3}'
    state_name = f'{cross_name}_state'
    
    # 获取双线交叉的状态列名
    cross1_2 = f'{ma1}_cross_{ma2}'
    cross2_3 = f'{ma2}_cross_{ma3}'
    state1_2 = f'{ma1}_cross_{ma2}_state'
    state2_3 = f'{ma2}_cross_{ma3}_state'
    
    # 检测多重交叉信号
    bullish_cross = (
        (dataframe[cross1_2] == 1) & (dataframe[state2_3] == 1) |
        (dataframe[state1_2] == 1) & (dataframe[cross2_3] == 1)
    )
    
    bearish_cross = (
        (dataframe[cross1_2] == 0) & (dataframe[state2_3] == 0) |
        (dataframe[state1_2] == 0) & (dataframe[cross2_3] == 0)
    )
    
    # 生成交叉信号
    dataframe[cross_name] = np.nan
    dataframe.loc[bullish_cross, cross_name] = 1
    dataframe.loc[bearish_cross, cross_name] = 0
    
    # 检测整体趋势状态
    bullish_state = (dataframe[state1_2] == 1) & (dataframe[state2_3] == 1)
    bearish_state = (dataframe[state1_2] == 0) & (dataframe[state2_3] == 0)
    
    # 生成状态信号
    dataframe[state_name] = np.nan
    dataframe.loc[bullish_state, state_name] = 1
    dataframe.loc[bearish_state, state_name] = 0
    dataframe[state_name] = dataframe[state_name].ffill()  # 填充状态
    
    return dataframe