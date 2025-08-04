import logging
from functools import reduce
from typing import Dict, Optional
import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.exchange.exchange_utils import *
from freqtrade.strategy import IStrategy, RealParameter, IntParameter
from freqtrade.persistence import Trade
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EnhancedLSTMStrategy(IStrategy):
    """
    Enhanced LSTM Strategy with Adaptive Leverage and Improved Exit Logic
    - Supports up to 100x leverage
    - Adaptive leverage based on market conditions and model confidence
    - Advanced trend persistence and exit logic
    - Reduced sensitivity to short-term market noise
    """
    # Hyperspace parameters:
    buy_params = {
        "threshold_buy": 0.59453,
        "w0": 0.54347,
        "w1": 0.82226,
        "w2": 0.56675,
        "w3": 0.77918,
        "w4": 0.98488,
        "w5": 0.31368,
        "w6": 0.75916,
        "w7": 0.09226,
        "w8": 0.85667,
        "leverage_multiplier": 1.0,        # 基础杠杆倍数，改为1.0支持更宽范围
        "min_leverage": 1,                 # 最小杠杆
        "max_leverage": 15,          # 最大杠杆
        "volatility_threshold": 0.02,
        "confidence_threshold": 0.7,
        "trend_persistence_bars": 3,
        "exit_confidence_threshold": 0.8,
        # 完全自适应资金管理参数 (1%-100%)
        "stake_multiplier": 0.8,           # 基础资金倍数
        "confidence_stake_factor": 2.0,    # 信心度资金因子
        "volatility_stake_factor": 1.5,    # 波动率资金因子
        "max_stake_ratio": 0.4,            # 最大资金使用比例
        "min_stake_ratio": 0.01,           # 最小资金使用比例1%
    }

    sell_params = {
        "threshold_sell": 0.80573,
        "risk_reduction_factor": 0.8,
        "trend_reversal_threshold": 0.75,
        "exit_smoothing_period": 5,
    }

    # ROI table - 适应高杠杆和大资金使用
    minimal_roi = {
        "600": 0     # Hold after 48 hours
    }

    # Stoploss - 固定止损，不做动态调整
    stoploss = -0.6

    # 跟踪止损
    trailing_stop = True  # 固定跟踪止损
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.02  # 基础跟踪止损偏移
    trailing_only_offset_is_reached = True
    use_custom_stoploss = False  # 自定义止损（基于杠杆）
    
    # 基础跟踪止损参数（将根据杠杆动态调整）
    base_trailing_stop_positive = 0.001
    base_trailing_stop_positive_offset = 0.0139

    timeframe = "1h"
    can_short = True
    use_exit_signal = True
    process_only_new_candles = True
    leverage_max = 100  # Maximum leverage

    startup_candle_count = 100

    # 无保护机制 - 完全由LSTM模型决定
    @property
    def protections(self):
        return []  # 移除所有保护机制

    # Enhanced strategy parameters
    threshold_buy = RealParameter(-1, 1, default=0.5, space='buy')
    threshold_sell = RealParameter(-1, 1, default=-0.5, space='sell')
    
    # Leverage parameters (支持1-100倍完整范围)
    leverage_multiplier = RealParameter(0.5, 2.0, default=1.0, space='buy')
    min_leverage = IntParameter(1, 5, default=1, space='buy')  # 最小杠杆改为1
    max_leverage = IntParameter(50, 100, default=100, space='buy')
    volatility_threshold = RealParameter(0.005, 0.05, default=0.02, space='buy')
    confidence_threshold = RealParameter(0.5, 0.95, default=0.7, space='buy')
    
    # Trend persistence parameters
    trend_persistence_bars = IntParameter(2, 8, default=3, space='buy')
    exit_confidence_threshold = RealParameter(0.6, 0.95, default=0.8, space='sell')
    trend_reversal_threshold = RealParameter(0.5, 0.9, default=0.75, space='sell')
    exit_smoothing_period = IntParameter(3, 10, default=5, space='sell')
    
    # Risk management
    risk_reduction_factor = RealParameter(0.5, 1.0, default=0.8, space='sell')
    
    # Feature switches (功能开关)
    enable_adaptive_leverage = False    # 自适应杠杆开关 (False = 使用固定杠杆) 有问题 会把杠杆吃满
    enable_adaptive_stake = False       # 自适应资金管理开关 (False = 使用config中的stake_amount)
    enable_percentage_stake = True     # 百分比资金模式开关 (True = 使用总资金的百分比)
    
    # Fixed parameters (when features are disabled)
    fixed_leverage = 15.0              # 固定杠杆倍数（当enable_adaptive_leverage=False时使用）
    percentage_stake_ratio = 0.2       # 总资金百分比 

    # Weights for calculating the aggregate score
    w0 = RealParameter(0, 1, default=0.10, space='buy')
    w1 = RealParameter(0, 1, default=0.15, space='buy')
    w2 = RealParameter(0, 1, default=0.10, space='buy')
    w3 = RealParameter(0, 1, default=0.15, space='buy')
    w4 = RealParameter(0, 1, default=0.10, space='buy')
    w5 = RealParameter(0, 1, default=0.10, space='buy')
    w6 = RealParameter(0, 1, default=0.10, space='buy')
    w7 = RealParameter(0, 1, default=0.05, space='buy')
    w8 = RealParameter(0, 1, default=0.15, space='buy')

    # 完全自适应资金管理参数 (支持1%-100%资金使用)
    stake_multiplier = RealParameter(0.3, 1.5, default=0.8, space='buy')  # 基础资金倍数
    confidence_stake_factor = RealParameter(1.0, 3.0, default=2.0, space='buy')  # 信心度资金因子
    volatility_stake_factor = RealParameter(0.5, 2.5, default=1.5, space='buy')  # 波动率资金因子
    max_stake_ratio = RealParameter(0.1, 0.3, default=0.3, space='buy')  # 最大资金使用比例30%
    min_stake_ratio = RealParameter(0.01, 0.1, default=0.01, space='buy')  # 最小资金使用比例

    def calculate_adaptive_leverage(self, dataframe: DataFrame, current_index: int) -> float:
        """
        基于LSTM模型信心度的智能杠杆计算
        主要基于confidence_smooth，在min_leverage到max_leverage范围内动态调整
        综合考虑信心度(70%)和目标强度(30%)，支持四舍五入
        """
        # 如果自适应杠杆关闭，返回固定杠杆
        if not self.enable_adaptive_leverage:
            return self.fixed_leverage
            
        try:
            if current_index < 20:  # Not enough data
                return self.min_leverage.value  # 默认最小杠杆
                
            # 获取信心度和目标强度
            confidence = dataframe['confidence_smooth'].iloc[current_index] if 'confidence_smooth' in dataframe.columns else 0.5
            target_strength = abs(dataframe['&-target'].iloc[current_index])
            
            # 综合信心度计算：主要基于confidence(70%)，target_strength作为调节(30%)
            combined_confidence = confidence * 0.7 + target_strength * 0.3
            
            # 确保combined_confidence在合理范围内(0-1)
            combined_confidence = max(0, min(1, combined_confidence))
            
            # 映射到杠杆范围：从min_leverage到max_leverage
            leverage_range = self.max_leverage.value - self.min_leverage.value
            leverage = self.min_leverage.value + (combined_confidence * leverage_range * self.leverage_multiplier.value)
            
            # 确保在范围内并四舍五入
            leverage = max(self.min_leverage.value, min(self.max_leverage.value, leverage))
            leverage = round(leverage)
            
            return leverage
            
        except Exception as e:
            logger.warning(f"Error calculating adaptive leverage: {e}")
            return self.min_leverage.value  # 错误时返回最小杠杆

    def calculate_position_size(self, dataframe: DataFrame, current_index: int, 
                              stake_amount: float, current_rate: float) -> float:
        """
        Calculate position size - this method is currently unused
        Note: In futures trading, position size is automatically calculated as:
        Position Size = (Stake Amount × Leverage) / Current Price
        But this is handled by the exchange, not by the strategy
        """
        # This method is not currently used - position sizing is handled by FreqTrade
        # based on stake_amount and leverage returned by respective methods
        return stake_amount / current_rate

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                       metadata: Dict, **kwargs):

        dataframe["%-cci-period"] = ta.CCI(dataframe, timeperiod=20)
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=10)
        dataframe["%-momentum-period"] = ta.MOM(dataframe, timeperiod=4)
        dataframe['%-ma-period'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['%-macd-period'], dataframe['%-macdsignal-period'], dataframe['%-macdhist-period'] = ta.MACD(
            dataframe['close'], slowperiod=12,
            fastperiod=26)
        dataframe['%-roc-period'] = ta.ROC(dataframe, timeperiod=2)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]
        dataframe["%-bb_width-period"] = (
                                                 dataframe["bb_upperband-period"]
                                                 - dataframe["bb_lowerband-period"]
                                         ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = (
                dataframe["close"] / dataframe["bb_lowerband-period"]
        )

        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict, **kwargs):

        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        
        # Add leverage-related features
        dataframe["%-volatility"] = dataframe["close"].pct_change().rolling(14).std()
        dataframe["%-trend_strength"] = abs(dataframe["close"].pct_change(10))
        
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: Dict, **kwargs):

        dataframe['date'] = pd.to_datetime(dataframe['date'])
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        
        # Add market session indicators
        dataframe["%-asian_session"] = ((dataframe["%-hour_of_day"] >= 0) & 
                                       (dataframe["%-hour_of_day"] < 8)).astype(int)
        dataframe["%-european_session"] = ((dataframe["%-hour_of_day"] >= 8) & 
                                          (dataframe["%-hour_of_day"] < 16)).astype(int)
        dataframe["%-us_session"] = ((dataframe["%-hour_of_day"] >= 16) & 
                                    (dataframe["%-hour_of_day"] < 24)).astype(int)
        
        return dataframe

    # 移除趋势持续性和平滑退出信号函数 - LSTM完全决定

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:

        dataframe['ma'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=2)
        dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = ta.MACD(dataframe['close'], slowperiod=12,
                                                                                    fastperiod=26)
        dataframe['momentum'] = ta.MOM(dataframe, timeperiod=4)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=10)
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_upperband'] = bollinger['upperband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_lowerband'] = bollinger['lowerband']
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        dataframe['stoch'] = ta.STOCH(dataframe)['slowk']
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['obv'] = ta.OBV(dataframe)

        # Enhanced indicators for trend analysis
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=14)
        
        # Add trend strength and persistence indicators
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['trend_strength'] = abs(dataframe['ema_21'] - dataframe['ema_50']) / dataframe['close']
        dataframe['price_momentum'] = dataframe['close'].pct_change(5)
        
        # Market structure indicators
        dataframe['higher_high'] = (dataframe['high'] > dataframe['high'].shift(1)) & (dataframe['high'].shift(1) > dataframe['high'].shift(2))
        dataframe['lower_low'] = (dataframe['low'] < dataframe['low'].shift(1)) & (dataframe['low'].shift(1) < dataframe['low'].shift(2))

        # Normalize indicators for leverage calculation
        dataframe['atr_normalized'] = (dataframe['atr'] / dataframe['close']).rolling(20).mean()
        dataframe['bb_width_normalized'] = ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) / 
                                           dataframe['bb_middleband']).rolling(20).mean()

        # Step 1: Normalize Indicators
        dataframe['normalized_stoch'] = (dataframe['stoch'] - dataframe['stoch'].rolling(window=14).mean()) / dataframe[
            'stoch'].rolling(window=14).std()
        dataframe['normalized_atr'] = (dataframe['atr'] - dataframe['atr'].rolling(window=14).mean()) / dataframe[
            'atr'].rolling(window=14).std()
        dataframe['normalized_obv'] = (dataframe['obv'] - dataframe['obv'].rolling(window=14).mean()) / dataframe[
            'obv'].rolling(window=14).std()
        dataframe['normalized_ma'] = (dataframe['close'] - dataframe['close'].rolling(window=10).mean()) / dataframe[
            'close'].rolling(window=10).std()
        dataframe['normalized_macd'] = (dataframe['macd'] - dataframe['macd'].rolling(window=26).mean()) / dataframe[
            'macd'].rolling(window=26).std()
        dataframe['normalized_roc'] = (dataframe['roc'] - dataframe['roc'].rolling(window=2).mean()) / dataframe[
            'roc'].rolling(window=2).std()
        dataframe['normalized_momentum'] = (dataframe['momentum'] - dataframe['momentum'].rolling(window=4).mean()) / \
                                           dataframe['momentum'].rolling(window=4).std()
        dataframe['normalized_rsi'] = (dataframe['rsi'] - dataframe['rsi'].rolling(window=10).mean()) / dataframe[
            'rsi'].rolling(window=10).std()
        dataframe['normalized_bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']).rolling(
            window=20).mean() / (dataframe['bb_upperband'] - dataframe['bb_lowerband']).rolling(window=20).std()
        dataframe['normalized_cci'] = (dataframe['cci'] - dataframe['cci'].rolling(window=20).mean()) / dataframe[
            'cci'].rolling(window=20).std()
        dataframe['normalized_adx'] = (dataframe['adx'] - dataframe['adx'].rolling(window=14).mean()) / dataframe[
            'adx'].rolling(window=14).std()

        # Dynamic Weights with leverage consideration
        trend_strength = abs(dataframe['ma'] - dataframe['close'])
        strong_trend_threshold = trend_strength.rolling(window=14).mean() + 1.5 * trend_strength.rolling(
            window=14).std()
        is_strong_trend = trend_strength > strong_trend_threshold

        # Enhanced dynamic weights
        dataframe['w_momentum'] = np.where(is_strong_trend, self.w3.value * 1.5, self.w3.value)
        dataframe['w_trend'] = np.where(dataframe['adx'] > 25, 1.2, 0.8)  # ADX > 25 indicates strong trend

        # Step 2: Enhanced aggregate score calculation
        w = [self.w0.value, self.w1.value, self.w2.value, self.w3.value, self.w4.value, self.w5.value,
             self.w6.value, self.w7.value, self.w8.value]

        dataframe['S'] = (w[0] * dataframe['normalized_ma'] + w[1] * dataframe['normalized_macd'] + 
                         w[2] * dataframe['normalized_roc'] + w[3] * dataframe['normalized_rsi'] + 
                         w[4] * dataframe['normalized_bb_width'] + w[5] * dataframe['normalized_cci'] + 
                         dataframe['w_momentum'] * dataframe['normalized_momentum'] + 
                         self.w8.value * dataframe['normalized_stoch'] + self.w7.value * dataframe['normalized_atr'] + 
                         self.w6.value * dataframe['normalized_obv'] + 0.1 * dataframe['normalized_adx'])

        # Step 3: Enhanced Market Regime Filter
        dataframe['R'] = 0
        dataframe.loc[(dataframe['close'] > dataframe['bb_middleband']) & (
                dataframe['close'] > dataframe['bb_upperband']), 'R'] = 1
        dataframe.loc[(dataframe['close'] < dataframe['bb_middleband']) & (
                dataframe['close'] < dataframe['bb_lowerband']), 'R'] = -1

        # Additional Market Regime Filter
        dataframe['ma_100'] = ta.SMA(dataframe, timeperiod=100)
        dataframe['R2'] = np.where(dataframe['close'] > dataframe['ma_100'], 1, -1)
        
        # Trend strength regime
        dataframe['R3'] = np.where(dataframe['adx'] > 25, 1.5, 0.8)

        # Step 4: Enhanced Volatility Adjustment
        bb_width = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']
        dataframe['V'] = 1 / (bb_width + 0.001)  # Prevent division by zero
        
        # ATR-based volatility
        dataframe['V2'] = 1 / (dataframe['atr_normalized'] + 0.001)
        
        # Volume-based confirmation
        dataframe['V3'] = np.where(dataframe['volume'] > dataframe['volume'].rolling(20).mean(), 1.2, 0.8)

        # Step 5: Final Target Score with leverage considerations
        dataframe['T'] = (dataframe['S'] * dataframe['R'] * dataframe['V'] * 
                         dataframe['R2'] * dataframe['V2'] * dataframe['R3'] * dataframe['V3'])

        # Enhanced confidence score for trend persistence
        dataframe['confidence'] = abs(dataframe['T']).rolling(8).mean()
        dataframe['confidence_smooth'] = dataframe['confidence'].rolling(3).mean()
        
        # Add trend reversal detection
        dataframe['trend_reversal_score'] = (
            abs(dataframe['T'].diff()) * 
            (dataframe['adx'] / 100) * 
            dataframe['trend_strength'] * 10
        ).rolling(5).mean()

        # Assign the target score T to the AI target column
        dataframe['&-target'] = dataframe['T']

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self.freqai_info = self.config["freqai"]

        dataframe = self.freqai.start(dataframe, metadata, self)
        
        # Add leverage calculation and exit signals after FreqAI processing
        for i in range(len(dataframe)):
            if i >= 20:  # Ensure we have enough data
                leverage = self.calculate_adaptive_leverage(dataframe, i)
                dataframe.loc[dataframe.index[i], 'calculated_leverage'] = leverage
                
                # 不再计算平滑退出信号 - LSTM直接决定
                dataframe.loc[dataframe.index[i], 'smooth_exit_signal'] = dataframe['&-target'].iloc[i]
            else:
                dataframe.loc[dataframe.index[i], 'calculated_leverage'] = self.min_leverage.value
                dataframe.loc[dataframe.index[i], 'smooth_exit_signal'] = 0
                
        return dataframe

    def calculate_adaptive_stake_ratio(self, dataframe: DataFrame, current_index: int) -> float:
        """
        LSTM模型完全决定仓位 - 无风险管理限制
        直接基于模型输出计算1%-100%资金使用
        如果enable_adaptive_stake为False，返回None表示使用固定stake
        """
        # 如果自适应资金管理关闭，返回None
        if not self.enable_adaptive_stake:
            return None
            
        try:
            if current_index < 20:
                return 0.5  # 默认50%
                
            # 直接使用LSTM模型输出
            target_strength = abs(dataframe['&-target'].iloc[current_index])
            confidence = dataframe['confidence_smooth'].iloc[current_index] if 'confidence_smooth' in dataframe.columns else 0.5
            
            # 简单线性映射：模型输出越强，仓位越大
            # 结合信心度和目标强度
            stake_ratio = target_strength * confidence * self.stake_multiplier.value
            
            # 确保在最小和最大比例范围内
            stake_ratio = max(self.min_stake_ratio.value, min(self.max_stake_ratio.value, stake_ratio))
            
            return stake_ratio
            
        except Exception as e:
            logger.warning(f"Error calculating adaptive stake ratio: {e}")
            return 0.5  # 错误时默认50%
            
    def custom_stake_amount(self, pair: str, current_time, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        """
        Fully adaptive stake amount calculation - 1% to 100%!
        支持三种模式:
        1. 固定金额模式 (enable_adaptive_stake=False, enable_percentage_stake=False): 使用config中的stake_amount
        2. 百分比模式 (enable_percentage_stake=True): 使用总资金的固定百分比
        3. 自适应模式 (enable_adaptive_stake=True): 根据市场条件动态调整资金比例
        """
        # 获取可用余额
        available_balance = self.wallets.get_free(self.config['stake_currency'])
        
        # 模式1: 百分比模式 - 使用总资金的固定百分比
        if self.enable_percentage_stake:
            calculated_stake = available_balance * self.percentage_stake_ratio
            
            # 应用系统限制
            if min_stake:
                calculated_stake = max(calculated_stake, min_stake)
            if max_stake:
                calculated_stake = min(calculated_stake, max_stake)
                
            logger.info(f"Percentage stake for {pair}: {calculated_stake:.2f} USDT "
                       f"({self.percentage_stake_ratio:.1%} of {available_balance:.2f} USDT)")
            return calculated_stake
        
        # 模式2: 如果自适应资金管理也关闭，使用config中的固定stake
        if not self.enable_adaptive_stake:
            return proposed_stake
        
        # 模式3: 自适应模式 - 根据市场条件动态调整资金比例    
        try:
            # Get the latest dataframe
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            
            if len(dataframe) < 20:
                # Not enough data, use minimal stake
                return max(available_balance * self.min_stake_ratio.value, min_stake or 0)
            
            # Calculate adaptive stake ratio (just like adaptive leverage)
            current_index = len(dataframe) - 1
            stake_ratio = self.calculate_adaptive_stake_ratio(dataframe, current_index)
            
            # Calculate stake amount
            calculated_stake = available_balance * stake_ratio
            
            # Apply system bounds
            if min_stake:
                calculated_stake = max(calculated_stake, min_stake)
            if max_stake:
                calculated_stake = min(calculated_stake, max_stake)
            
            # Get market info for logging
            target_strength = abs(dataframe['&-target'].iloc[-1])
            confidence = dataframe['confidence_smooth'].iloc[-1] if 'confidence_smooth' in dataframe.columns else 0.5
            calculated_leverage = dataframe['calculated_leverage'].iloc[-1] if 'calculated_leverage' in dataframe.columns else 1
            
            logger.info(f"Adaptive stake for {pair}: {calculated_stake:.2f} USDT "
                       f"({stake_ratio:.1%} of {available_balance:.2f} USDT) - "
                       f"Target: {target_strength:.3f}, Confidence: {confidence:.3f}, "
                       f"Leverage: {calculated_leverage}x")
            
            return calculated_stake
            
        except Exception as e:
            logger.warning(f"Error in adaptive stake calculation: {e}")
            # Fallback to minimal risk
            try:
                available_balance = self.wallets.get_free(self.config['stake_currency'])
                fallback_stake = max(available_balance * self.min_stake_ratio.value, min_stake or 0)
                return min(fallback_stake, max_stake) if max_stake else fallback_stake
            except:
                return proposed_stake


    def leverage(self, pair: str, current_time, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Dynamic leverage calculation based on market conditions
        支持1-100倍完整范围
        如果enable_adaptive_leverage为False，返回固定杠杆
        """
        # 如果自适应杠杆关闭，返回固定杠杆
        if not self.enable_adaptive_leverage:
            # 从config中获取固定杠杆，如果没有则使用类属性
            fixed_lev = self.config.get('leverage', self.fixed_leverage)
            return min(fixed_lev, max_leverage)
            
        try:
            # Get the latest dataframe
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            
            if len(dataframe) < 20:
                return min(self.min_leverage.value, max_leverage)
                
            # Get calculated leverage
            calculated_leverage = dataframe['calculated_leverage'].iloc[-1]
            
            # Ensure it's within bounds
            final_leverage = min(calculated_leverage, max_leverage, self.max_leverage.value)
            final_leverage = max(final_leverage, 1)
            
            logger.info(f"LSTM leverage for {pair}: {final_leverage}x")
            
            return final_leverage
            
        except Exception as e:
            logger.warning(f"Error calculating leverage: {e}")
            return min(self.min_leverage.value, max_leverage)

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Enhanced entry conditions with stronger confirmation
        enter_long_conditions = [
            df["do_predict"] == 1,
            df['&-target'] > self.threshold_buy.value,
            df['volume'] > 0,
        ]

        enter_short_conditions = [
            df["do_predict"] == 1,
            df['&-target'] < self.threshold_sell.value,
            df["volume"] > 0,
        ]

        df.loc[
            reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
        ] = (1, "long_adaptive")

        df.loc[
            reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
        ] = (1, "short_adaptive")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # LSTM模型完全决定退出 - 无风险管理
        
        # 多头退出条件 - 仅基于LSTM信号
        exit_long_conditions = [
            df["do_predict"] == 1,
            df['&-target'] < self.threshold_sell.value,  # LSTM说卖就卖
        ]

        # 空头退出条件 - 仅基于LSTM信号
        exit_short_conditions = [
            df["do_predict"] == 1,
            df['&-target'] > self.threshold_buy.value,  # LSTM说买就平空
        ]

        if exit_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, exit_long_conditions), ["exit_long", "exit_tag"]
            ] = (1, "exit_long_lstm")

        if exit_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, exit_short_conditions), ["exit_short", "exit_tag"]
            ] = (1, "exit_short_lstm")

        return df

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, after_fill: bool,
                       **kwargs) -> float | None:
        """
        杠杆感知的跟踪止损 - 根据杠杆同比例放大跟踪止损参数
        杠杆越高, 跟踪止损距离越宽松, 避免高杠杆时被小波动止损
        """
        try:
            # 获取当前交易的杠杆
            leverage = trade.leverage or 1.0

            # 根据杠杆同比例放大跟踪止损参数
            # 杠杆越高, 止损距离越宽松
            adjusted_trailing_positive = self.base_trailing_stop_positive * leverage
            adjusted_trailing_offset = self.base_trailing_stop_positive_offset * leverage

            # 只有当盈利超过调整后的偏移量时才启动跟踪止损
            if current_profit > adjusted_trailing_offset:
                return -adjusted_trailing_positive  # 返回负数

            # 盈利不足, 不启动跟踪止损
            return None

        except Exception as e:
            return -self.base_trailing_stop_positive  # 返回负数
