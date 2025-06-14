# To customize the tkinter window adjust lines1467 - 1520


import yfinance as yf
import numpy as np
import pandas as pd
import ta
import ta.volatility as vol
import ta.momentum as mom
import ta.trend as trend
import ta.volume as volume
import warnings
import tkinter as tk
from tkinter import ttk
import os
import random

# Define the get_color_for_ticker function earlier in the script
import random

# Define a fixed color map at the top (before the loop)
if 'TICKER_COLORS' not in globals():
    TICKER_COLORS = {}
COLOR_POOL = [
    "#FF5733", "#33FF57", "#3357FF", "#F39C12", "#9B59B6", "#1ABC9C", "#E74C3C", "#2ECC71", "#3498DB"
]

def get_color_for_ticker(ticker):
    if ticker not in TICKER_COLORS:
        # Assign the next color from the pool or generate a new one
        if COLOR_POOL:
            TICKER_COLORS[ticker] = COLOR_POOL.pop(0)
        else:
            TICKER_COLORS[ticker] = "#%06x" % random.randint(0, 0xFFFFFF)
    return TICKER_COLORS[ticker]


def safe_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

# Suppress warnings
warnings.filterwarnings('ignore')


def identify_trends(df):
    """
    Identify short-term and long-term trends.
    """
    df['Short_Term_Trend_SMA'] = np.where(df['SMA_5'] > df['SMA_10'], 'Uptrend', 'Downtrend')
    df['Long_Term_Trend_SMA'] = np.where(df['SMA_50'] > df['SMA_200'], 'Uptrend', 'Downtrend')
    df['Short_Term_Trend_EMA'] = np.where(df['EMA_10'] > df['EMA_20'], 'Uptrend', 'Downtrend')
    return df


def get_growth_metrics(ticker):
    """
    Fetch growth metrics from Yahoo Finance.
    """
    stock = yf.Ticker(ticker)
    try:
        # Fetch quarterly earnings growth (year-over-year)
        quarterly_earnings_growth = stock.info.get('earningsQuarterlyGrowth', None)
  
        
        # Fetch quarterly revenue growth (year-over-year)
        quarterly_revenue_growth = stock.info.get('revenueGrowth', None)
        
        # Fetch annual earnings growth (year-over-year)
        annual_earnings_growth = stock.info.get('earningsGrowth', None)
        
        # Fetch profit margin
        profit_margin = stock.info.get('profitMargins', None)
        
        # Fetch trailing P/E ratio
        trailing_pe = stock.info.get('trailingPE', None)
        
        # Fetch forward P/E ratio
        forward_pe = stock.info.get('forwardPE', None)
        
        # Fetch quarterly financial data
        quarterly_financials = stock.quarterly_financials.T
        
        # Extract earnings, sales, and profit margins for the last 4 quarters
        earnings = quarterly_financials['Net Income'].head(4).tolist()
        sales = quarterly_financials['Total Revenue'].head(4).tolist()
        profit_margins = (quarterly_financials['Net Income'] / quarterly_financials['Total Revenue']).head(4).tolist()
        
        return {
            'Quarterly Earnings Growth (yoy)': quarterly_earnings_growth,
            'Quarterly Revenue Growth (yoy)': quarterly_revenue_growth,
            'Annual Earnings Growth (yoy)': annual_earnings_growth,
            'Profit Margin': profit_margin,
            'Trailing P/E': trailing_pe,
            'Forward P/E': forward_pe,
            'Earnings Last 4 Quarters': earnings,
            'Sales Last 4 Quarters': sales,
            'Profit Margins Last 4 Quarters': profit_margins
        }
    except Exception as e:
        return {
            'Quarterly Earnings Growth (yoy)': None,
            'Quarterly Revenue Growth (yoy)': None,
            'Annual Earnings Growth (yoy)': None,
            'Profit Margin': None,
            'Trailing P/E': None,
            'Forward P/E': None,
            'Earnings Last 4 Quarters': None,
            'Sales Last 4 Quarters': None,
            'Profit Margins Last 4 Quarters': None,
            'error': str(e)
        }
    


def classify_growth_metrics(growth_metrics):
    classifications = {}

    # Quarterly Earnings Growth (yoy)
    if growth_metrics['Quarterly Earnings Growth (yoy)'] is not None:
        if growth_metrics['Quarterly Earnings Growth (yoy)'] > 0.10:
            classifications['Quarterly Earnings Growth (yoy)'] = 'Bullish'
        elif growth_metrics['Quarterly Earnings Growth (yoy)'] < 0.05:
            classifications['Quarterly Earnings Growth (yoy)'] = 'Bearish'
        else:
            classifications['Quarterly Earnings Growth (yoy)'] = 'Neutral'
    else:
        classifications['Quarterly Earnings Growth (yoy)'] = 'N/A'

    # Quarterly Revenue Growth (yoy)
    if growth_metrics['Quarterly Revenue Growth (yoy)'] is not None:
        if growth_metrics['Quarterly Revenue Growth (yoy)'] > 0.10:
            classifications['Quarterly Revenue Growth (yoy)'] = 'Bullish'
        elif growth_metrics['Quarterly Revenue Growth (yoy)'] < 0.05:
            classifications['Quarterly Revenue Growth (yoy)'] = 'Bearish'
        else:
            classifications['Quarterly Revenue Growth (yoy)'] = 'Neutral'
    else:
        classifications['Quarterly Revenue Growth (yoy)'] = 'N/A'

    # Annual Earnings Growth (yoy)
    if growth_metrics['Annual Earnings Growth (yoy)'] is not None:
        if growth_metrics['Annual Earnings Growth (yoy)'] > 0.10:
            classifications['Annual Earnings Growth (yoy)'] = 'Bullish'
        elif growth_metrics['Annual Earnings Growth (yoy)'] < 0.05:
            classifications['Annual Earnings Growth (yoy)'] = 'Bearish'
        else:
            classifications['Annual Earnings Growth (yoy)'] = 'Neutral'
    else:
        classifications['Annual Earnings Growth (yoy)'] = 'N/A'

    # Profit Margin
    if growth_metrics['Profit Margin'] is not None:
        if growth_metrics['Profit Margin'] > 0.15:
            classifications['Profit Margin'] = 'Bullish'
        elif growth_metrics['Profit Margin'] < 0.05:
            classifications['Profit Margin'] = 'Bearish'
        else:
            classifications['Profit Margin'] = 'Neutral'
    else:
        classifications['Profit Margin'] = 'N/A'

    # Trailing P/E
    trailing_pe = safe_float(growth_metrics['Trailing P/E'])
    if trailing_pe is not None:
        if trailing_pe < 15:
            classifications['Trailing P/E'] = 'Bullish'
        elif trailing_pe > 25:
            classifications['Trailing P/E'] = 'Bearish'
        else:
            classifications['Trailing P/E'] = 'Neutral'
    else:
        classifications['Trailing P/E'] = 'N/A'

    # Forward P/E
    forward_pe = safe_float(growth_metrics['Forward P/E'])
    if forward_pe is not None:
        if forward_pe < 15:
            classifications['Forward P/E'] = 'Bullish'
        elif forward_pe > 25:
            classifications['Forward P/E'] = 'Bearish'
        else:
            classifications['Forward P/E'] = 'Neutral'
    else:
        classifications['Forward P/E'] = 'N/A'

    return classifications


def identify_consolidation(df):
    """
    Identify consolidation periods.
    """
    df['Consolidation'] = np.where((df['High'].rolling(window=20).max() - df['Low'].rolling(window=20).min()) / df['Close'] < 0.05, True, False)
    return df

def identify_potential_breakout(df):
    """
    Identify potential breakout and specify if it's bullish or bearish.
    """
    df['Potential_Breakout'] = np.where((df['Close'] > df['High'].rolling(window=20).max()), 'Bullish', 
                                        np.where((df['Close'] < df['Low'].rolling(window=20).min()), 'Bearish', False))
    return df



def calculate_aroon_oscillator(df):
    """
    Calculate Aroon Oscillator.
    """
    aroon = trend.AroonIndicator(df['High'], df['Low'])
    df['Aroon_Oscillator'] = aroon.aroon_indicator()
    return df

def calculate_smi(df):
    """
    Calculate Stochastic Momentum Index (SMI).
    """
    smi = mom.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['SMI'] = smi.stoch()
    return df

def calculate_gmma(df):
    """
    Calculate Guppy Multiple Moving Averages (GMMA).
    """
    short_emas = [df['Close'].ewm(span=n, adjust=False).mean() for n in [3, 5, 8, 10, 12, 15]]
    long_emas = [df['Close'].ewm(span=n, adjust=False).mean() for n in [30, 35, 40, 45, 50, 60]]
    for i, ema in enumerate(short_emas + long_emas):
        df[f'GMMA_{i+1}'] = ema
    return df

def calculate_vix(df):
    """
    Calculate Volatility Index (VIX).
    """
    df['VIX'] = vol.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    return df

def calculate_elders_impulse(df):
    """
    Calculate Elder's Impulse System.
    """
    ema = df['Close'].ewm(span=13, adjust=False).mean()
    macd = trend.MACD(df['Close']).macd()
    df['Elder_Impulse'] = np.where((df['Close'] > ema) & (macd > 0), 'Bullish', 
                                   np.where((df['Close'] < ema) & (macd < 0), 'Bearish', 'Neutral'))
    return df

def calculate_mfi(df):
    """
    Calculate Money Flow Index (MFI).
    """
    df['MFI'] = volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume']).money_flow_index()
    return df

def calculate_keltner_channels(df):
    """
    Calculate Keltner Channels.
    """
    keltner = vol.KeltnerChannel(df['High'], df['Low'], df['Close'])
    df['Keltner_High'] = keltner.keltner_channel_hband()
    df['Keltner_Low'] = keltner.keltner_channel_lband()
    return df

def calculate_donchian_channels(df):
    """
    Calculate Donchian Channels.
    """
    donchian = vol.DonchianChannel(df['High'], df['Low'], df['Close'])
    df['Donchian_High'] = donchian.donchian_channel_hband()
    df['Donchian_Low'] = donchian.donchian_channel_lband()
    return df


# Add this function to calculate ADR
def calculate_adr(df, period=14):
    """
    Calculate Average Daily Range (ADR).
    """
    df['Daily_Range'] = df['High'] - df['Low']
    df['ADR'] = df['Daily_Range'].rolling(window=period).mean()
    return df


def calculate_aroon(df):
    """
    Calculate Aroon Indicator.
    """
    aroon = ta.trend.AroonIndicator(df['High'], df['Low'])
    df['Aroon_Up'] = aroon.aroon_up()
    df['Aroon_Down'] = aroon.aroon_down()
    return df



def calculate_roc(df):
    """
    Calculate Rate of Change (ROC).
    """
    df['ROC'] = ta.momentum.ROCIndicator(df['Close']).roc()
    return df

def calculate_kst(df):
    """
    Calculate Know Sure Thing (KST).
    """
    kst = ta.trend.KSTIndicator(df['Close'])
    df['KST'] = kst.kst()
    df['KST_Signal'] = kst.kst_sig()
    return df

def calculate_force_index(df):
    """
    Calculate Force Index.
    """
    df['Force_Index'] = ta.volume.ForceIndexIndicator(df['Close'], df['Volume']).force_index()
    return df

def calculate_chaikin_money_flow(df):
    """
    Calculate Chaikin Money Flow.
    """
    df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume']).chaikin_money_flow()
    return df

def calculate_supertrend(df):
    """
    Calculate Supertrend Indicator.
    """
    supertrend = ta.trend.STCIndicator(df['Close'])
    df['Supertrend'] = supertrend.stc()
    return df




def calculate_rsi_divergence(df):
    """
    Calculate RSI divergence.
    """
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['RSI_Divergence'] = np.nan

    for i in range(1, len(df) - 1):
        if df['RSI'].iloc[i] < df['RSI'].iloc[i - 1] and df['RSI'].iloc[i] < df['RSI'].iloc[i + 1] and df['Close'].iloc[i] > df['Close'].iloc[i - 1] and df['Close'].iloc[i] > df['Close'].iloc[i + 1]:
            df['RSI_Divergence'].iloc[i] = 'Bullish'
        elif df['RSI'].iloc[i] > df['RSI'].iloc[i - 1] and df['RSI'].iloc[i] > df['RSI'].iloc[i + 1] and df['Close'].iloc[i] < df['Close'].iloc[i - 1] and df['Close'].iloc[i] < df['Close'].iloc[i + 1]:
            df['RSI_Divergence'].iloc[i] = 'Bearish'

    return df


def calculate_macd_divergence(df):
    """
    Calculate MACD divergence.
    """
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_Divergence'] = np.nan

    for i in range(1, len(df) - 1):
        if df['MACD'].iloc[i] < df['MACD'].iloc[i - 1] and df['MACD'].iloc[i] < df['MACD'].iloc[i + 1] and df['Close'].iloc[i] > df['Close'].iloc[i - 1] and df['Close'].iloc[i] > df['Close'].iloc[i + 1]:
            df['MACD_Divergence'].iloc[i] = 'Bullish'
        elif df['MACD'].iloc[i] > df['MACD'].iloc[i - 1] and df['MACD'].iloc[i] > df['MACD'].iloc[i + 1] and df['Close'].iloc[i] < df['Close'].iloc[i - 1] and df['Close'].iloc[i] < df['Close'].iloc[i + 1]:
            df['MACD_Divergence'].iloc[i] = 'Bearish'

    return df


def analyze_volume(df):
    """
    Analyze volume spikes.
    """
    df['Volume_Spike'] = df['Volume'] > df['Volume'].rolling(window=20).mean() * 1.5
    return df

def calculate_heikin_ashi(df):
    """
    Calculate Heikin-Ashi candles.
    """
    df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df['HA_Open'] = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    df['HA_High'] = df[['High', 'HA_Open', 'HA_Close']].max(axis=1)
    df['HA_Low'] = df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)
    return df

def calculate_pivot_points(df):
    """
    Calculate daily pivot points.
    """
    df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    df['R1'] = 2 * df['Pivot'] - df['Low'].shift(1)
    df['S1'] = 2 * df['Pivot'] - df['High'].shift(1)
    df['R2'] = df['Pivot'] + (df['High'].shift(1) - df['Low'].shift(1))
    df['S2'] = df['Pivot'] - (df['High'].shift(1) - df['Low'].shift(1))
    df['R3'] = df['High'].shift(1) + 2 * (df['Pivot'] - df['Low'].shift(1))
    df['S3'] = df['Low'].shift(1) - 2 * (df['High'].shift(1) - df['Pivot'])
    return df




def advanced_technical_analysis(df):
    """
    Perform comprehensive technical analysis on stock data.
    Parameters:
    - df: DataFrame with stock price data
    Returns:
    - DataFrame with technical indicators
    """
    # Trend Indicators
    df['SMA_5'] = df['Close'].rolling(window=5).mean()  
    df['SMA_10'] = df['Close'].rolling(window=10).mean()  
    df['SMA_20'] = df['Close'].rolling(window=20).mean()  # Add this line
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_150'] = df['Close'].rolling(window=150).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_33'] = df['Close'].ewm(span=33, adjust=False).mean()  # Add this line
    df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # Add a new column to indicate if the close price is above the 9 EMA
    df['Close_Above_EMA_9'] = df['Close'] > df['EMA_9']

    # Ichimoku Cloud Components
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    df['Tenkan_Sen'] = (high_9 + low_9) / 2
    df['Kijun_Sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2

    # Momentum Indicators
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['Stoch_RSI'] = ta.momentum.StochRSIIndicator(df['Close']).stochrsi()
    df['Williams_R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()

    # Volatility Indicators
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_high'] = bollinger.bollinger_hband()
    df['BB_low'] = bollinger.bollinger_lband()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()

    # Trend and Momentum Indicators
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()

    # On-Balance Volume
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

    # Trend Strength
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
    df['ADX'] = adx.adx()
    df['+DI'] = adx.adx_pos()
    df['-DI'] = adx.adx_neg()

    # Short-term ADX calculation
    short_term_adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
    df['Short_Term_ADX'] = short_term_adx.adx()
    df['Short_Term_+DI'] = short_term_adx.adx_pos()
    df['Short_Term_-DI'] = short_term_adx.adx_neg()

    # Parabolic SAR
    df['Parabolic_SAR'] = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close']).psar()

    # VWAP
    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()

    # Commodity Channel Index (CCI)
    df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()

    # Accumulation/Distribution Line (ADL)
    df['ADL'] = ta.volume.AccDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume']).acc_dist_index()

    # Additional indicators
    df = calculate_adr(df)
    df = calculate_rsi_divergence(df)
    df = calculate_macd_divergence(df)
    df = analyze_volume(df)
    df = calculate_heikin_ashi(df)
    df = calculate_pivot_points(df)
    df = calculate_aroon(df)
    df = calculate_roc(df)
    df = calculate_kst(df)
    df = calculate_force_index(df)
    df = calculate_chaikin_money_flow(df)
    df = calculate_supertrend(df)
    df = calculate_aroon_oscillator(df)
    df = calculate_smi(df)
    df = calculate_gmma(df)
    df = calculate_vix(df)
    df = calculate_elders_impulse(df)
    df = calculate_mfi(df)
    df = calculate_keltner_channels(df)
    df = calculate_donchian_channels(df)
    # Identify trends, consolidation, and potential breakout
    df = identify_trends(df)
    df = identify_consolidation(df)
    df = identify_potential_breakout(df)

    return df




def identify_candlestick_patterns(df):
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

    def is_doji(row, tolerance=0.1):
        body_size = abs(row['Close'] - row['Open'])
        total_range = row['High'] - row['Low']
        return body_size <= total_range * tolerance

    def is_hammer(row):
        body_size = abs(row['Close'] - row['Open'])
        total_range = row['High'] - row['Low']
        lower_shadow = min(row['Open'], row['Close']) - row['Low']
        return (lower_shadow >= total_range * 0.6) and (body_size <= total_range * 0.3)

    def is_shooting_star(row):
        body_size = abs(row['Close'] - row['Open'])
        total_range = row['High'] - row['Low']
        upper_shadow = row['High'] - max(row['Open'], row['Close'])
        return (upper_shadow >= total_range * 0.6) and (body_size <= total_range * 0.3)

    def is_engulfing(current, previous):
        bullish_engulfing = (
            previous['Close'] < previous['Open'] and
            current['Close'] > current['Open'] and
            current['Close'] > previous['High'] and
            current['Open'] < previous['Low']
        )
        bearish_engulfing = (
            previous['Close'] > previous['Open'] and
            current['Close'] < current['Open'] and
            current['Close'] < previous['Low'] and
            current['Open'] > previous['High']
        )
        return bullish_engulfing, bearish_engulfing

    def is_morning_star(df):
        return (df.iloc[-3]['Close'] < df.iloc[-3]['Open'] and
                abs(df.iloc[-2]['Close'] - df.iloc[-2]['Open']) < (df.iloc[-2]['High'] - df.iloc[-2]['Low']) * 0.3 and
                df.iloc[-1]['Close'] > df.iloc[-1]['Open'] and
                df.iloc[-1]['Close'] > (df.iloc[-3]['Close'] + df.iloc[-3]['Open']) / 2)

    def is_evening_star(df):
        return (df.iloc[-3]['Close'] > df.iloc[-3]['Open'] and
                abs(df.iloc[-2]['Close'] - df.iloc[-2]['Open']) < (df.iloc[-2]['High'] - df.iloc[-2]['Low']) * 0.3 and
                df.iloc[-1]['Close'] < df.iloc[-1]['Open'] and
                df.iloc[-1]['Close'] < (df.iloc[-3]['Close'] + df.iloc[-3]['Open']) / 2)

    def is_harami(current, previous):
        bullish_harami = (
            previous['Close'] < previous['Open'] and
            current['Close'] > current['Open'] and
            current['High'] <= previous['Open'] and
            current['Low'] >= previous['Close']
        )
        bearish_harami = (
            previous['Close'] > previous['Open'] and
            current['Close'] < current['Open'] and
            current['High'] <= previous['Close'] and
            current['Low'] >= previous['Open']
        )
        return bullish_harami, bearish_harami

    def is_piercing_line(current, previous):
        return (
            previous['Close'] < previous['Open'] and
            current['Open'] < previous['Close'] and
            current['Close'] > previous['Open'] and
            current['Close'] >= (previous['Low'] + previous['High']) / 2
        )

    def is_dark_cloud_cover(current, previous):
        return (
            previous['Close'] > previous['Open'] and
            current['Open'] > previous['Close'] and
            current['Close'] < previous['Low'] and
            current['Close'] <= (previous['Low'] + previous['High']) / 2
        )

    def is_three_white_soldiers(df):
        return (
            len(df) >= 3 and
            all(df.iloc[-i-1]['Close'] > df.iloc[-i-1]['Open'] for i in range(3)) and
            all(df.iloc[-i-1]['Close'] > df.iloc[-i-2]['Close'] for i in range(1, 3))
        )

    def is_three_black_crows(df):
        return (
            len(df) >= 3 and
            all(df.iloc[-i-1]['Close'] < df.iloc[-i-1]['Open'] for i in range(3)) and
            all(df.iloc[-i-1]['Close'] < df.iloc[-i-2]['Close'] for i in range(1, 3))
        )

    def is_spinning_top(row, tolerance=0.3):
        body_size = abs(row['Close'] - row['Open'])
        total_range = row['High'] - row['Low']
        upper_shadow = row['High'] - max(row['Open'], row['Close'])
        lower_shadow = min(row['Open'], row['Close']) - row['Low']
        return (body_size <= total_range * tolerance and
                upper_shadow >= body_size and
                lower_shadow >= body_size)

    def is_marubozu(row, tolerance=0.05):
        body_size = abs(row['Close'] - row['Open'])
        total_range = row['High'] - row['Low']
        if body_size >= total_range * (1 - tolerance):
            if row['Close'] > row['Open']:
                return 'bullish'
            else:
                return 'bearish'
        return None

    def is_tweezer_top(current, previous):
        return (
            previous['Close'] > previous['Open'] and
            current['Close'] < current['Open'] and
            abs(previous['High'] - current['High']) <= (previous['High'] - previous['Low']) * 0.1
        )

    def is_tweezer_bottom(current, previous):
        return (
            previous['Close'] < previous['Open'] and
            current['Close'] > current['Open'] and
            abs(previous['Low'] - current['Low']) <= (previous['High'] - previous['Low']) * 0.1
        )

    def is_inside_bar(current, previous):
        return (
            current['High'] <= previous['High'] and
            current['Low'] >= previous['Low']
        )

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    patterns = {
        'doji': {
            'detected': is_doji(latest),
            'type': 'Neutral',
            'description': f"{YELLOW}(Neutral) : Indicates market indecision or potential trend reversal (Short-term){RESET}"
        },
        'hammer': {
            'detected': is_hammer(latest),
            'type': 'Bullish',
            'description': f"{GREEN}(Bullish) : Suggests potential bullish reversal after a downtrend (Short-term){RESET}"
        },
        'shooting_star': {
            'detected': is_shooting_star(latest),
            'type': 'Bearish',
            'description': f"{RED}(Bearish) : Indicates potential bearish reversal after an uptrend (Short-term){RESET}"
        },
        'bullish_engulfing': {
            'detected': is_engulfing(latest, previous)[0],
            'type': 'Bullish',
            'description': f"{GREEN}(Bullish) : Strong bullish reversal pattern showing potential trend change (Short-term){RESET}"
        },
        'bearish_engulfing': {
            'detected': is_engulfing(latest, previous)[1],
            'type': 'Bearish',
            'description': f"{RED}(Bearish) : Strong bearish reversal pattern showing potential trend change (Short-term){RESET}"
        },
        'morning_star': {
            'detected': is_morning_star(df),
            'type': 'Bullish',
            'description': f"{GREEN}(Bullish) : Bullish reversal pattern indicating a potential uptrend (Short-term){RESET}"
        },
        'evening_star': {
            'detected': is_evening_star(df),
            'type': 'Bearish',
            'description': f"{RED}(Bearish) : Bearish reversal pattern indicating a potential downtrend (Short-term){RESET}"
        },
        'bullish_harami': {
            'detected': is_harami(latest, previous)[0],
            'type': 'Bullish',
            'description': f"{GREEN}(Bullish) : Bullish reversal pattern suggesting a potential uptrend (Short-term){RESET}"
        },
        'bearish_harami': {
            'detected': is_harami(latest, previous)[1],
            'type': 'Bearish',
            'description': f"{RED}(Bearish) : Bearish reversal pattern suggesting a potential downtrend (Short-term){RESET}"
        },
        'piercing_line': {
            'detected': is_piercing_line(latest, previous),
            'type': 'Bullish',
            'description': f"{GREEN}(Bullish) : Bullish reversal pattern indicating a potential uptrend after a downtrend (Short-term){RESET}"
        },
        'dark_cloud_cover': {
            'detected': is_dark_cloud_cover(latest, previous),
            'type': 'Bearish',
            'description': f"{RED}(Bearish) : Bearish reversal pattern indicating a potential downtrend after an uptrend (Short-term){RESET}"
        },
        'three_white_soldiers': {
            'detected': is_three_white_soldiers(df),
            'type': 'Bullish',
            'description': f"{GREEN}(Bullish) : Bullish continuation pattern indicating strong buying pressure (Short-term){RESET}"
        },
        'three_black_crows': {
            'detected': is_three_black_crows(df),
            'type': 'Bearish',
            'description': f"{RED}(Bearish) : Bearish continuation pattern indicating strong selling pressure (Short-term){RESET}"
        },
        'spinning_top': {
            'detected': is_spinning_top(latest),
            'type': 'Neutral',
            'description': f"{YELLOW}(Neutral) : Indicates indecision between buyers and sellers (Short-term){RESET}"
        },
        'marubozu': {
            'detected': is_marubozu(latest),
            'type': 'Bullish' if is_marubozu(latest) == 'bullish' else 'Bearish',
            'description': f"Indicates strong conviction in the current trend ({GREEN if is_marubozu(latest) == 'bullish' else RED}{is_marubozu(latest)}{RESET}) (Short-term)" if is_marubozu(latest) else "Indicates strong conviction in the current trend (Short-term)"
        },
        'tweezer_top': {
            'detected': is_tweezer_top(latest, previous),
            'type': 'Bearish',
            'description': f"{RED}(Bearish) : Potential bearish reversal pattern at the top of an uptrend (Short-term){RESET}"
        },
        'tweezer_bottom': {
            'detected': is_tweezer_bottom(latest, previous),
            'type': 'Bullish',
            'description': f"{GREEN}(Bullish) : Potential bullish reversal pattern at the bottom of a downtrend (Short-term){RESET}"
        },
        'inside_bar': {
            'detected': is_inside_bar(latest, previous),
            'type': 'Neutral',
            'description': f"{YELLOW}(Neutral) : Indicates consolidation and potential continuation or reversal (Short-term){RESET}"
        }
    }

    return patterns






def calculate_fibonacci_retracement(df):
    """
    Calculate Fibonacci retracement levels.
    """
    max_price = df['High'].max()
    min_price = df['Low'].min()
    diff = max_price - min_price


    levels = {
        'level_0': max_price,
        'level_23.60': max_price - 0.236 * diff,
        'level_38.20': max_price - 0.382 * diff,
        'level_50.00': max_price - 0.5 * diff,
        'level_61.80': max_price - 0.618 * diff,
        'level_78.60': max_price - 0.786 * diff,
        'level_100': min_price
    }

    return levels






def calculate_support_resistance(df):
    """
    Calculate support and resistance levels.
    """
    support = df['Low'].rolling(window=20).min().iloc[-1]
    resistance = df['High'].rolling(window=20).max().iloc[-1]
    return support, resistance




def generate_comprehensive_signal(df, latest, patterns, fibonacci_levels):
    """
    Generate comprehensive trading signal with detailed explanations.
    """
    try:
        bullish_score = 0
        bearish_score = 0
        neutral_score = 0
        short_term_bullish_score = 0
        short_term_bearish_score = 0
        long_term_bullish_score = 0
        long_term_bearish_score = 0
        short_term_bullish_reasons = []
        short_term_bearish_reasons = []
        long_term_bullish_reasons = []
        long_term_bearish_reasons = []
        neutral_reasons = []

        # 1. Moving Average Analysis
        if latest['SMA_50'] > latest['SMA_200']:
            bullish_score += 2
            long_term_bullish_score += 2
            long_term_bullish_reasons.append("50-day MA is above 200-day MA (Bullish Long-term Trend)")
        else:
            bearish_score += 2
            long_term_bearish_score += 2
            long_term_bearish_reasons.append("50-day MA is below 200-day MA (Bearish Long-term Trend)")


        if latest['SMA_5'] > latest['SMA_10']:
            bullish_score += 1
            short_term_bullish_score += 1
            short_term_bullish_reasons.append("5-day MA is above 10-day MA (Bullish Short-term Trend)")
        else:
            bearish_score += 1
            short_term_bearish_score += 1
            short_term_bearish_reasons.append("5-day MA is below 10-day MA (Bearish Short-term Trend)")

        # Add checks for 5-day and 10-day EMAs
        if latest['EMA_5'] > latest['EMA_10']:
            bullish_score += 1
            short_term_bullish_score += 1
            short_term_bullish_reasons.append("5-day EMA is above 10-day EMA (Bullish Short-term Trend)")
        else:
            bearish_score += 1
            short_term_bearish_score += 1
            short_term_bearish_reasons.append("5-day EMA is below 10-day EMA (Bearish Short-term Trend)")
        


        # Crossover Analysis
        if latest['SMA_50'] > latest['SMA_200'] and df['SMA_50'].iloc[-2] <= df['SMA_200'].iloc[-2]:
            bullish_score += 3
            long_term_bullish_score += 3
            long_term_bullish_reasons.append("Golden Cross detected (Bullish - 50-day MA crossed above 200-day MA) (Long-term)")
        elif latest['SMA_50'] < latest['SMA_200'] and df['SMA_50'].iloc[-2] >= df['SMA_200'].iloc[-2]:
            bearish_score += 3
            long_term_bearish_score += 3
            long_term_bearish_reasons.append("Death Cross detected (Bearish - 50-day MA crossed below 200-day MA) (Long-term)")

        # 2. RSI Analysis
        if latest['RSI'] < 30:
            bullish_score += 2
            short_term_bullish_score += 2
            short_term_bullish_reasons.append("RSI below 30 (Bullish - Oversold Condition, Potential Bounce) (Short-term)")
        elif latest['RSI'] > 70:
            bearish_score += 2
            short_term_bearish_score += 2
            short_term_bearish_reasons.append("RSI above 70 (Bearish - Overbought Condition, Potential Pullback) (Short-term)")

        # 3. MACD Analysis
        if latest['MACD'] > latest['MACD_signal']:
            bullish_score += 2
            short_term_bullish_score += 2
            short_term_bullish_reasons.append("MACD Line above Signal Line (Bullish Momentum) (Short-term)")
        else:
            bearish_score += 2
            short_term_bearish_score += 2
            short_term_bearish_reasons.append("MACD Line below Signal Line (Bearish Momentum) (Short-term)")

        # 4. Bollinger Bands Analysis
        if latest['Close'] < latest['BB_low']:
            bullish_score += 1
            short_term_bullish_score += 1
            short_term_bullish_reasons.append("Price near Lower Bollinger Band (Potential Bullish Reversal) (Short-term)")
        elif latest['Close'] > latest['BB_high']:
            bearish_score += 1
            short_term_bearish_score += 1
            short_term_bearish_reasons.append("Price near Upper Bollinger Band (Potential Bearish Reversal) (Short-term)")

        # 5. Candlestick Patterns
        for pattern, info in patterns.items():
            if info['detected']:
                if 'bullish' in pattern:
                    bullish_score += 3
                    short_term_bullish_score += 3
                    short_term_bullish_reasons.append(f"{pattern.replace('_', ' ').title()} Detected: {info['description']} (Short-term)")
                elif 'bearish' in pattern:
                    bearish_score += 3
                    short_term_bearish_score += 3
                    short_term_bearish_reasons.append(f"{pattern.replace('_', ' ').title()} Detected: {info['description']} (Short-term)")
                elif 'indecision' in pattern:
                    neutral_score += 3
                    neutral_reasons.append(f"{pattern.replace('_', ' ').title()} Detected: {info['description']} (Neutral)")

        # 6. ADX Trend Strength
        if latest['ADX'] > 25:
            if latest['+DI'] > latest['-DI']:
                bullish_score += 2
                long_term_bullish_score += 2
                long_term_bullish_reasons.append("Strong Bullish Trend Confirmed by ADX (Long-term)")
            else:
                bearish_score += 2
                long_term_bearish_score += 2
                long_term_bearish_reasons.append("Strong Bearish Trend Confirmed by ADX (Long-term)")

        if latest['Short_Term_ADX'] > 25:
            if latest['Short_Term_+DI'] > latest['Short_Term_-DI']:
                bullish_score += 1
                short_term_bullish_score += 1
                short_term_bullish_reasons.append("Strong Bullish Trend Confirmed by Short-Term ADX")
            else:
                bearish_score += 1
                short_term_bearish_score += 1
                short_term_bearish_reasons.append("Strong Bearish Trend Confirmed by Short-Term ADX")

        # 7. Volume and Momentum
        if latest['OBV'] > df['OBV'].rolling(window=10).mean().iloc[-1]:
            bullish_score += 1
            short_term_bullish_score += 1
            short_term_bullish_reasons.append("On-Balance Volume Increasing (Bullish Volume Trend) (Short-term)")
        else:
            bearish_score += 1
            short_term_bearish_score += 1
            short_term_bearish_reasons.append("On-Balance Volume Decreasing (Bearish Volume Trend) (Short-term)")

        # 8. MACD Histogram Analysis
        if latest['MACD_hist'] > 0:
            bullish_score += 1
            short_term_bullish_score += 1
            short_term_bullish_reasons.append("MACD Histogram is positive (Bullish Momentum) (Short-term)")
        else:
            bearish_score += 1
            short_term_bearish_score += 1
            short_term_bearish_reasons.append("MACD Histogram is negative (Bearish Momentum) (Short-term)")

        # 9. Parabolic SAR Analysis
        if latest['Close'] > latest['Parabolic_SAR']:
            bullish_score += 1
            short_term_bullish_score += 1
            short_term_bullish_reasons.append("Price is above Parabolic SAR (Bullish) (Short-term)")
        else:
            bearish_score += 1
            short_term_bearish_score += 1
            short_term_bearish_reasons.append("Price is below Parabolic SAR (Bearish) (Short-term)")

        # 10. VWAP Analysis
        if latest['Close'] > latest['VWAP']:
            bullish_score += 1
            short_term_bullish_score += 1
            short_term_bullish_reasons.append("Price is above VWAP (Bullish) (Short-term)")
        else:
            bearish_score += 1
            short_term_bearish_score += 1
            short_term_bearish_reasons.append("Price is below VWAP (Bearish) (Short-term)")

        # 10-day EMA Analysis
        if latest['Close'] > latest['EMA_10']:
            bullish_score += 1
            short_term_bullish_score += 1
            short_term_bullish_reasons.append("Price is above 10-day EMA (Bullish - Short-term Trend)")
        else:
            bearish_score += 1
            short_term_bearish_score += 1
            short_term_bearish_reasons.append("Price is below 10-day EMA (Bearish - Short-term Trend)")

        # 20-day EMA Analysis
        if latest['Close'] > latest['EMA_20']:
            bullish_score += 1
            short_term_bullish_score += 1
            short_term_bullish_reasons.append("Price is above 20-day EMA (Bullish - Short-term Trend)")
        else:
            bearish_score += 1
            short_term_bearish_score += 1
            short_term_bearish_reasons.append("Price is below 20-day EMA (Bearish - Short-term Trend)")

        # 100-day EMA Analysis
        if latest['Close'] > latest['EMA_100']:
            bullish_score += 1
            long_term_bullish_score += 1
            long_term_bullish_reasons.append("Price is above 100-day EMA (Bullish - Long-term Trend)")
        else:
            bearish_score += 1
            long_term_bearish_score += 1
            long_term_bearish_reasons.append("Price is below 100-day EMA (Bearish - Long-term Trend)")

        # 14. EMA 200 Analysis
        if latest['Close'] > latest['EMA_200']:
            bullish_score += 2
            long_term_bullish_score += 2
            long_term_bullish_reasons.append("Price is above 200-day EMA (Bullish Long-term Trend)")
        else:
            bearish_score += 2
            long_term_bearish_score += 2
            long_term_bearish_reasons.append("Price is below 200-day EMA (Bearish Long-term Trend)")

        # 11. Stochastic Oscillator Analysis
        if latest['Stoch_K'] > latest['Stoch_D']:
            bullish_score += 1
            short_term_bullish_score += 1
            short_term_bullish_reasons.append("Stochastic %K is above %D (Bullish Momentum) (Short-term)")
        else:
            bearish_score += 1
            short_term_bearish_score += 1
            short_term_bearish_reasons.append("Stochastic %K is below %D (Bearish Momentum) (Short-term)")

        if latest['Stoch_K'] < 20:
            bullish_score += 1
            short_term_bullish_score += 1
            short_term_bullish_reasons.append("Stochastic %K is below 20 (Bullish - Oversold Condition) (Short-term)")
        elif latest['Stoch_K'] > 80:
            bearish_score += 1
            short_term_bearish_score += 1
            short_term_bearish_reasons.append("Stochastic %K is above 80 (Bearish - Overbought Condition) (Short-term)")

        # 12. CCI Analysis
        if latest['CCI'] < -100:
            bullish_score += 1
            short_term_bullish_score += 1
            short_term_bullish_reasons.append("CCI below -100 (Bullish - Oversold Condition, potential for upward reversal) (Short-term)")
        elif latest['CCI'] > 100:
            bearish_score += 1
            short_term_bearish_score += 1
            short_term_bearish_reasons.append("CCI above 100 (Bearish - Overbought Condition, potential for downward reversal) (Short-term)")
            short_term_bullish_reasons.append("CCI above 100 (Bullish - Strong Uptrend, indicating strong upward momentum) (Short-term)")

        # 13. Fibonacci Retracement Levels Analysis
        for level, value in fibonacci_levels.items():
            if latest['Close'] > value:
                bullish_score += 1
                long_term_bullish_score += 1
                short_term_bullish_score += 1  # Add this line for short-term analysis
                long_term_bullish_reasons.append(f"Price above Fibonacci {level} (Bullish - indicating potential support) (Long-term)")
                short_term_bullish_reasons.append(f"Price above Fibonacci {level} (Bullish - indicating potential support) (Short-term)")  
            else:
                bearish_score += 1
                long_term_bearish_score += 1
                short_term_bearish_score += 1  # Add this line for short-term analysis
                long_term_bearish_reasons.append(f"Price below Fibonacci {level} (Bearish - indicating potential resistance) (Long-term)")
                short_term_bearish_reasons.append(f"Price below Fibonacci {level} (Bearish - indicating potential resistance) (Short-term)")  

        

        # 15. Accumulation/Distribution Line (ADL) Analysis
        if latest['ADL'] > df['ADL'].rolling(window=50).mean().iloc[-1]:
            bullish_score += 1
            long_term_bullish_score += 1
            long_term_bullish_reasons.append("ADL is above its 50-day average (Bullish Long-term Trend)")
        else:
            bearish_score += 1
            long_term_bearish_score += 1
            long_term_bearish_reasons.append("ADL is below its 50-day average (Bearish Long-term Trend)")

        # Short-term analysis
        if latest['ADL'] > df['ADL'].rolling(window=10).mean().iloc[-1]:
            bullish_score += 1
            short_term_bullish_score += 1
            short_term_bullish_reasons.append("ADL is above its 10-day average (Bullish Short-term Trend)")
        else:
            bearish_score += 1
            short_term_bearish_score += 1
            short_term_bearish_reasons.append("ADL is below its 10-day average (Bearish Short-term Trend)")

        # Support and Resistance Levels Analysis
        support, resistance = calculate_support_resistance(df)
        if latest['Close'] > resistance:
            bullish_score += 2
            long_term_bullish_score += 2
            short_term_bullish_score += 2  # Add this line for short-term analysis
            long_term_bullish_reasons.append("Price broke above resistance level (Bullish) (Long-term)")
            short_term_bullish_reasons.append("Price broke above resistance level (Bullish) (Short-term)")  
        elif latest['Close'] < support:
            bearish_score += 2
            long_term_bearish_score += 2
            short_term_bearish_score += 2  # Add this line for short-term analysis
            long_term_bearish_reasons.append("Price broke below support level (Bearish) (Long-term)")
            short_term_bearish_reasons.append("Price broke below support level (Bearish) (Short-term)")  

        # Additional Indicators Analysis
        # RSI Divergence Analysis
        if latest['RSI_Divergence'] == 'Bullish':
            bullish_score += 2
            short_term_bullish_score += 2
            short_term_bullish_reasons.append("RSI Divergence (Bullish) (Short-term)")
        elif latest['RSI_Divergence'] == 'Bearish':
            bearish_score += 2
            short_term_bearish_score += 2
            short_term_bearish_reasons.append("RSI Divergence (Bearish) (Short-term)")

        # MACD Divergence Analysis
        if latest['MACD_Divergence'] == 'Bullish':
            bullish_score += 2
            short_term_bullish_score += 2
            short_term_bullish_reasons.append("MACD Divergence (Bullish) (Short-term)")
        elif latest['MACD_Divergence'] == 'Bearish':
            bearish_score += 2
            short_term_bearish_score += 2
            short_term_bearish_reasons.append("MACD Divergence (Bearish) (Short-term)")

        # Volume Spike Analysis
        if latest['Volume_Spike']:
            bullish_score += 1
            short_term_bullish_score += 1
            short_term_bullish_reasons.append("Volume Spike Detected (Bullish) (Short-term)")

        # Heikin-Ashi Analysis
        if latest['HA_Close'] > latest['HA_Open']:
            bullish_score += 1
            short_term_bullish_score += 1
            short_term_bullish_reasons.append("Heikin-Ashi Candle (Bullish) (Short-term)")
        else:
            bearish_score += 1
            short_term_bearish_score += 1
            short_term_bearish_reasons.append("Heikin-Ashi Candle (Bearish) (Short-term)")

        # Pivot Points Analysis
        if latest['Close'] > latest['Pivot']:
            bullish_score += 1
            short_term_bullish_score += 1
            short_term_bullish_reasons.append("Price above Pivot Point (Bullish) (Short-term)")
        else:
            bearish_score += 1
            short_term_bearish_score += 1
            short_term_bearish_reasons.append("Price below Pivot Point (Bearish) (Short-term)")


        

        # Aroon Indicator Analysis
        if latest['Aroon_Up'] > latest['Aroon_Down']:
            bullish_score += 1
            short_term_bullish_score += 1
            short_term_bullish_reasons.append("Aroon Up is above Aroon Down (Bullish) (Short-term)")
        else:
            bearish_score += 1
            short_term_bearish_score += 1
            short_term_bearish_reasons.append("Aroon Down is above Aroon Up (Bearish) (Short-term)")

        # ROC Analysis
        if latest['ROC'] > 0:
            bullish_score += 1
            short_term_bullish_score += 1
            short_term_bullish_reasons.append("ROC is positive (Bullish Momentum) (Short-term)")
        else:
            bearish_score += 1
            short_term_bearish_score += 1
            short_term_bearish_reasons.append("ROC is negative (Bearish Momentum) (Short-term)")

        # KST Analysis
        if latest['KST'] > latest['KST_Signal']:
            bullish_score += 1
            short_term_bullish_score += 1
            short_term_bullish_reasons.append("KST is above Signal Line (Bullish Momentum) (Short-term)")
        else:
            bearish_score += 1
            short_term_bearish_score += 1
            short_term_bearish_reasons.append("KST is below Signal Line (Bearish Momentum) (Short-term)")

        # Force Index Analysis
        if latest['Force_Index'] > 0:
            bullish_score += 1
            short_term_bullish_score += 1
            short_term_bullish_reasons.append("Force Index is positive (Bullish) (Short-term)")
        else:
            bearish_score += 1
            short_term_bearish_score += 1
            short_term_bearish_reasons.append("Force Index is negative (Bearish) (Short-term)")

        # Chaikin Money Flow Analysis
        if latest['CMF'] > 0:
            bullish_score += 1
            short_term_bullish_score += 1
            short_term_bullish_reasons.append("Chaikin Money Flow is positive (Bullish) (Short-term)")
        else:
            bearish_score += 1
            short_term_bearish_score += 1
            short_term_bearish_reasons.append("Chaikin Money Flow is negative (Bearish) (Short-term)")

        # Supertrend Analysis
        if latest['Supertrend'] > 0:
            bullish_score += 1
            short_term_bullish_score += 1
            short_term_bullish_reasons.append("Supertrend is positive (Bullish) (Short-term)")
        else:
            bearish_score += 1
            short_term_bearish_score += 1
            short_term_bearish_reasons.append("Supertrend is negative (Bearish) (Short-term)")

        # Final Signal Determination
        if bullish_score > bearish_score and bullish_score > neutral_score:
            trend = 'Bullish'
            confidence = min(bullish_score / (bullish_score + bearish_score + neutral_score) * 100, 100)
            primary_reasons = short_term_bullish_reasons + long_term_bullish_reasons
        elif bearish_score > bullish_score and bearish_score > neutral_score:
            trend = 'Bearish'
            confidence = min(bearish_score / (bullish_score + bearish_score + neutral_score) * 100, 100)
            primary_reasons = short_term_bearish_reasons + long_term_bearish_reasons
        else:
            trend = 'Neutral'
            confidence = min(neutral_score / (bullish_score + bearish_score + neutral_score) * 100, 100)
            primary_reasons = neutral_reasons if neutral_reasons else ["Conflicting Signals", "No Clear Market Direction"]

        short_term_trend = 'Bullish' if short_term_bullish_score > short_term_bearish_score else 'Bearish'
        short_term_confidence = min(short_term_bullish_score / (short_term_bullish_score + short_term_bearish_score) * 100, 100)
        short_term_bearish_confidence = min(short_term_bearish_score / (short_term_bullish_score + short_term_bearish_score) * 100, 100)
        short_term_bullish_confidence = min(short_term_bullish_score / (short_term_bullish_score + short_term_bearish_score) * 100, 100)

        long_term_trend = 'Bullish' if long_term_bullish_score > long_term_bearish_score else 'Bearish'
        long_term_confidence = min(long_term_bullish_score / (long_term_bullish_score + long_term_bearish_score) * 100, 100)
        long_term_bearish_confidence = min(long_term_bearish_score / (long_term_bullish_score + long_term_bearish_score) * 100, 100)
        long_term_bullish_confidence = min(long_term_bullish_score / (long_term_bullish_score + long_term_bearish_score) * 100, 100)

        return {
            'trend': trend,
            'confidence': confidence,
            'primary_reasons': primary_reasons,
            'short_term_trend': short_term_trend,
            'short_term_confidence': short_term_confidence,
            'short_term_bullish_confidence': short_term_bullish_confidence,
            'short_term_bearish_confidence': short_term_bearish_confidence,
            'short_term_reasons': short_term_bullish_reasons if short_term_trend == 'Bullish' else short_term_bearish_reasons,
            'long_term_trend': long_term_trend,
            'long_term_confidence': long_term_confidence,
            'long_term_bullish_confidence': long_term_bullish_confidence,
            'long_term_bearish_confidence': long_term_bearish_confidence,
            'long_term_reasons': long_term_bullish_reasons if long_term_trend == 'Bullish' else long_term_bearish_reasons
        }
    except Exception as e:
        return {"error": str(e)}             






def analyze_stock(ticker, period='1y', interval='1d'):
    """
    Comprehensive stock analysis function.
    """
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            return {"error": "Unable to fetch stock data"}

        # Perform advanced technical analysis
        df = advanced_technical_analysis(df)

        # Identify candlestick patterns
        patterns = identify_candlestick_patterns(df)

        # Get latest data point
        latest = df.iloc[-1]

        # Calculate Fibonacci retracement levels
        fibonacci_levels = calculate_fibonacci_retracement(df)

        # Generate comprehensive signal
        signal = generate_comprehensive_signal(df, latest, patterns, fibonacci_levels)

        # Calculate support and resistance levels
        support, resistance = calculate_support_resistance(df)

        # Fetch growth metrics
        growth_metrics = get_growth_metrics(ticker)

        technical_indicators = {
            "ADX": latest['ADX'],
            "+DI": latest['+DI'],
            "-DI": latest['-DI'],
            "ADL": latest['ADL'],
            "ADR": latest['ADR'],
            "ATR": latest['ATR'],  
            "Aroon_Down": latest['Aroon_Down'],
            "Aroon_Up": latest['Aroon_Up'],
            "BB_high": latest['BB_high'],
            "BB_low": latest['BB_low'],
            "CCI": latest['CCI'],
            "CMF": latest['CMF'],
            "Consolidation": latest['Consolidation'],
            "EMA_5": latest['EMA_5'],
            "EMA_9": latest['EMA_9'],
            "EMA_10": latest['EMA_10'],
            "EMA_20": latest['EMA_20'],
            "EMA_100": latest['EMA_100'],
            "EMA_200": latest['EMA_200'],
            "Force_Index": latest['Force_Index'],
            "HA_Close": latest['HA_Close'],
            "HA_Open": latest['HA_Open'],
            "KST": latest['KST'],
            "KST_Signal": latest['KST_Signal'],
            "Long_Term_Trend_SMA": latest['Long_Term_Trend_SMA'],
            "MACD": latest['MACD'],
            "MACD_Divergence": latest['MACD_Divergence'],
            "MACD_signal": latest['MACD_signal'],
            "Pivot": latest['Pivot'],
            "Potential_Breakout": latest['Potential_Breakout'],
            "Resistance": resistance,
            "ROC": latest['ROC'],
            "RSI": latest['RSI'],
            "RSI_Divergence": latest['RSI_Divergence'],
            "SMA_5": latest['SMA_5'],
            "SMA_10": latest['SMA_10'],
            "SMA_50": latest['SMA_50'],
            "SMA_150": latest['SMA_150'],
            "SMA_200": latest['SMA_200'],
            "Short_Term_Trend_EMA": latest['Short_Term_Trend_EMA'],
            "Short_Term_Trend_SMA": latest['Short_Term_Trend_SMA'],
            "Stoch_D": latest['Stoch_D'],
            "Stoch_K": latest['Stoch_K'],
            "Support": support,
            "Supertrend": latest['Supertrend'],
            "Volume_Spike": latest['Volume_Spike'],
            "Quarterly Earnings Growth (yoy)": growth_metrics['Quarterly Earnings Growth (yoy)'],
            "Annual Earnings Growth (yoy)": growth_metrics['Annual Earnings Growth (yoy)'],
            "Quarterly Revenue Growth (yoy)": growth_metrics['Quarterly Revenue Growth (yoy)'],
            "Profit Margin": growth_metrics['Profit Margin'],
            "Trailing P/E": growth_metrics['Trailing P/E'],
            "Forward P/E": growth_metrics['Forward P/E'],
            "Earnings Last 4 Quarters": growth_metrics['Earnings Last 4 Quarters'],
            "Sales Last 4 Quarters": growth_metrics['Sales Last 4 Quarters'],
            "Profit Margins Last 4 Quarters": growth_metrics['Profit Margins Last 4 Quarters']
        }

        return {
            "ticker": ticker,
            "latest_price": latest['Close'],
            "volume": latest['Volume'],
            "technical_indicators": technical_indicators,
            "candlestick_patterns": patterns,
            "signal": signal,
            "fibonacci_levels": fibonacci_levels,
            "full_data": df
        }
    except Exception as e:
        return {"error": str(e)}
    
def calculate_atr(df, period=14):
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df['ATR']

def interpret_atr(atr_value, avg_atr, std_atr):
    high_threshold = avg_atr + std_atr
    low_threshold = avg_atr - std_atr

    if atr_value > high_threshold:
        return "High Volatility"
    elif atr_value < low_threshold:
        return "Low Volatility"
    else:
        return "Moderate Volatility"


def refresh_data(tree, tickers, period, interval):
    results = []

    for ticker in tickers:
        analysis = analyze_stock(ticker, period=period, interval=interval)
        if 'error' in analysis:
            print(f"Error analyzing {ticker}: {analysis['error']}")
            continue

        signal = analysis['signal']
        latest = analysis['full_data'].iloc[-1]
        growth_metrics = analysis['technical_indicators']
        # Add previous row for trend comparison
        previous = analysis['full_data'].iloc[-2] if len(analysis['full_data']) > 1 else latest

        # Calculate ATR
        atr_series = calculate_atr(analysis['full_data'])
        avg_atr = atr_series.mean()
        std_atr = atr_series.std()
        latest_atr = atr_series.iloc[-1]

        # Interpret ATR
        atr_interpretation = interpret_atr(latest_atr, avg_atr, std_atr)

        # Classify growth metrics
        growth_classifications = classify_growth_metrics(growth_metrics)

        def get_trend(value1, value2):
            return "Bullish" if value1 > value2 else "Bearish"

        def get_volume_trend(volume, avg_volume):
            if volume > avg_volume * 1.5:
                return "High"
            elif volume < avg_volume * 0.5:
                return "Low"
            else:
                return "Average"

        avg_volume = analysis['full_data']['Volume'].rolling(window=20).mean().iloc[-1]
        volume_trend = get_volume_trend(latest['Volume'], avg_volume)
        latest_volume = latest['Volume']

        # Calculate support and resistance levels
        support, resistance = calculate_support_resistance(analysis['full_data'])

        # Debug prints to check the values
        print(f"Ticker: {ticker}")
        print(f"Latest Close: {latest['Close']}")
        print(f"Support: {support}")
        print(f"Resistance: {resistance}")

        # Determine if resistance and support levels are bullish or bearish
        if latest['Close'] > resistance:
            resistance_trend = "Bullish Breakout"
            support_trend = "Not close to support line"  # No need to check support if breakout over resistance
        elif latest['Close'] < support:
            support_trend = "Bearish Breakout"
            resistance_trend = "Not close to resistance line"  # No need to check resistance if breakout under support
        else:
            # Determine which level the price is closer to
            if abs(latest['Close'] - resistance) < abs(latest['Close'] - support):
                if abs(latest['Close'] - resistance) / resistance < 0.05:
                    resistance_trend = "Bearish (Possible Bouncing off Resistance)"
                    support_trend = "Not close to support line"
                else:
                    resistance_trend = "Not close to resistance line"
                    support_trend = "Not close to support line"
            else:
                if abs(latest['Close'] - support) / support < 0.05:
                    support_trend = "Bullish (Possible Bouncing off Support)"
                    resistance_trend = "Not close to resistance line"
                else:
                    resistance_trend = "Not close to resistance line"
                    support_trend = "Not close to support line"

        # Debug prints to check the trends
        print(f"Resistance Trend: {resistance_trend}")
        print(f"Support Trend: {support_trend}")

        # Calculate the percentage difference to the support and resistance lines
        percentage_to_support = ((latest['Close'] - support) / support) * 100
        percentage_to_resistance = ((resistance - latest['Close']) / resistance) * 100

        # --- RSI and MACD trend comparison ---
        if latest['RSI'] > previous['RSI']:
            rsi_trend = "Rising"
        elif latest['RSI'] < previous['RSI']:
            rsi_trend = "Falling"
        else:
            rsi_trend = "Unchanged"

        if latest['MACD'] > previous['MACD']:
            macd_trend = "Rising"
        elif latest['MACD'] < previous['MACD']:
            macd_trend = "Falling"
        else:
            macd_trend = "Unchanged"
        # Determine if high volume is bullish or bearish
        prev_close = analysis['full_data']['Close'].iloc[-2]

        if volume_trend == "High":
            if latest['Close'] > prev_close:
                volume_bullish_bearish = "Bullish"
            else:
                volume_bullish_bearish = "Bearish"
        elif volume_trend == "Low":
            if latest['Close'] > prev_close:
                volume_bullish_bearish = "Weak Bullish"
            else:
                volume_bullish_bearish = "Weak Bearish"
        else:
            volume_bullish_bearish = "Neutral"

        # Calculate VWAP
        vwap = (latest['Close'] * latest['Volume']) / latest['Volume']

        # Determine if the signal is bullish or bearish based on VWAP
        vwap_signal = "Bullish" if latest['Close'] > latest['VWAP'] else "Bearish"

        # Extract earnings data
        earnings = growth_metrics['Earnings Last 4 Quarters']

        # Determine the trend of earnings for the last 4 quarters
        if earnings and len(earnings) >= 4:
            earnings_trend = "Bullish" if all(earnings[i] < earnings[i + 1] for i in range(len(earnings) - 1)) else "Bearish" if all(earnings[i] > earnings[i + 1] for i in range(len(earnings) - 1)) else "Neutral"
        else:
            earnings_trend = "N/A"

        # Extract sales data
        sales = growth_metrics.get('Sales Last 4 Quarters')

        # Determine the trend of sales for the last 4 quarters
        if sales and len(sales) >= 4:
            sales_trend = "Bullish" if all(sales[i] < sales[i + 1] for i in range(len(sales) - 1)) else "Bearish" if all(sales[i] > sales[i + 1] for i in range(len(sales) - 1)) else "Neutral"
        else:
            sales_trend = "N/A"

        # Extract profit margins data
        profit_margins = growth_metrics.get('Profit Margins Last 4 Quarters')

        # Determine the trend of profit margins for the last 4 quarters
        if profit_margins and len(profit_margins) >= 4:
            profit_margins_trend = "Bullish" if all(profit_margins[i] < profit_margins[i + 1] for i in range(len(profit_margins) - 1)) else "Bearish" if all(profit_margins[i] > profit_margins[i + 1] for i in range(len(profit_margins) - 1)) else "Neutral"
        else:
            profit_margins_trend = "N/A"

        # Convert P/E ratios to float
        trailing_pe = safe_float(growth_metrics['Trailing P/E'])
        forward_pe = safe_float(growth_metrics['Forward P/E'])

        # Calculate Fibonacci retracement levels
        fibonacci_levels = calculate_fibonacci_retracement(analysis['full_data'])

        # Determine the overall trend using shorter-term EMAs
        overall_trend = 'Bullish' if latest['EMA_9'] > latest['EMA_20'] else 'Bearish'
        
        # Determine if the price is bouncing off a Fibonacci level
        def determine_fibonacci_trend(level):
            if overall_trend == 'Bullish':
                return 'Support' if latest['Close'] > fibonacci_levels[level] else 'Resistance'
            else:
                return 'Resistance' if latest['Close'] < fibonacci_levels[level] else 'Support'
        
        # Determine the closest Fibonacci levels
        def closest_fibonacci_levels():
            levels = sorted(fibonacci_levels.items(), key=lambda x: x[1])
            below = next((level for level in reversed(levels) if level[1] < latest['Close']), None)
            above = next((level for level in levels if level[1] > latest['Close']), None)
            return below, above
        
        below_level, above_level = closest_fibonacci_levels()
        
        result = {
            'ticker': ticker,
            'overall_trend': signal['trend'],
            'overall_trend_confidence': f"{signal['confidence']:.2f}%",
            'volume_trend': volume_trend,
            'volume_trend_signal': volume_bullish_bearish,
            'latest_price': f"${analysis['latest_price']:.2f}" + (" (Whole Number)" if analysis['latest_price'].is_integer() else ""),
            'trend_based_on_levels': (
                'Bullish (Possible Breakout over resistance line)' if latest['Close'] > resistance else
                'Bearish (Possible Breakout under support line)' if latest['Close'] < support else
                'Bearish (Possible Bouncing off Resistance, check other indicators for possible breakthrough)' if abs(latest['Close'] - resistance) / resistance < 0.05 else
                'Bullish (Possible Bouncing off Support)' if abs(latest['Close'] - support) / support < 0.05 else
                'Neutral (Not close to either support or resistance line)'
            ),
            'short_term_bullish_trend_confidence': f"{signal['short_term_bullish_confidence']:.2f}%",
            'Price_Above_9_20_33_200_EMAs': (
                "Bullish" if (latest['Close'] > latest['EMA_9'] and
                              latest['EMA_9'] > latest['EMA_20'] and
                              latest['EMA_20'] > latest['EMA_33'] and
                              latest['EMA_33'] > latest['EMA_200']) else "Bearish"
            ),
            'long_term_bullish_trend_confidence': f"{signal['long_term_bullish_confidence']:.2f}%",
            '20_SMA_Above_200_SMA (Long-term)': "Bullish" if latest['SMA_20'] > latest['SMA_200'] else "Bearish",
            'Price_Above_50_150_200_SMA_and_SMA_Alignment': (
                "Bullish" if (latest['Close'] > latest['SMA_50'] and
                              latest['Close'] > latest['SMA_150'] and
                              latest['Close'] > latest['SMA_200'] and
                              latest['SMA_50'] > latest['SMA_150'] and
                              latest['SMA_50'] > latest['SMA_200'] and
                              latest['SMA_150'] > latest['SMA_200'])
                else "Bearish"
            ),
            'MACD': "Bullish" if latest['MACD'] > latest['MACD_signal'] else "Bearish",
            'RSI': f"{'Buy Signal' if latest['RSI'] < 30 else 'Sell Signal' if latest['RSI'] > 70 else 'Neutral)'} ({latest['RSI']:.2f})",
            'Bollinger_Low': "Buy Signal" if latest['Close'] <= latest['BB_low'] else "Neutral",
            'Bollinger_High': "Sell Signal" if latest['Close'] >= latest['BB_high'] else "Neutral",
            'Aroon_Down': "Bearish" if latest['Aroon_Down'] > latest['Aroon_Up'] else "Neutral",
            'Aroon_Up': "Bullish" if latest['Aroon_Up'] > latest['Aroon_Down'] else "Neutral",
            'Stoch_K': (
                "Buy Signal" if latest['Stoch_K'] < 20 else
                "Sell Signal" if latest['Stoch_K'] > 80 else
                "Neutral"
            ),
            'Stoch_D': (
                "Buy Signal" if latest['Stoch_K'] > latest['Stoch_D'] and latest['Stoch_K'] < 20 and latest['Stoch_D'] < 20 else
                "Sell Signal" if latest['Stoch_K'] < latest['Stoch_D'] and latest['Stoch_K'] > 80 and latest['Stoch_D'] > 80 else
                "Neutral"
            ),
            'ADX': f"{'Strong' if latest['ADX'] > 25 else 'Weak'} {'Bullish' if latest['+DI'] > latest['-DI'] else 'Bearish'} Trend ({latest['ADX']:.2f})",
            'ATR': f"{latest['ATR']:.2f} ({atr_interpretation})",
            'VWAP': f"${latest['VWAP']:.2f} ({vwap_signal})",
            'Candlestick_Pattern': next(
                (f"{pattern} ({info['type']})"
                for pattern, info in analysis['candlestick_patterns'].items() if info['detected']), 'None'),
            'Fibonacci_Levels': {
                'trend_direction': 'Uptrend' if overall_trend == 'Bullish' else 'Downtrend',
                'closest_below': {
                    'level': f"{below_level[0].split('_')[1]}%",
                    'value': f"${below_level[1]:.2f}",
                    'trend': determine_fibonacci_trend(below_level[0])
                } if below_level else 'None',
                'closest_above': {
                    'level': f"{above_level[0].split('_')[1]}%",
                    'value': f"${above_level[1]:.2f}",
                    'trend': determine_fibonacci_trend(above_level[0])
                } if above_level else 'None'
            }
        }
        # Calculate the bullish score
        bullish_score = sum(1 for key, value in result.items() if isinstance(value, str) and 'Bullish' in value)
        result['Bullish_Score'] = bullish_score
        result['MACD_Trend'] = macd_trend
        result['RSI_Trend'] = rsi_trend
        
        results.append(result)

    # Write results to CSV file 
    results_df = pd.DataFrame(results)
    # Filter results_df to only include tickers from stocks_to_analyze.csv before saving
    stocks_to_analyze = pd.read_csv('../stock_csv_files/stocks_to_analyze.csv')
    valid_tickers = set(stocks_to_analyze['ticker'].astype(str).str.upper())
    # Try both 'ticker' and 'Ticker' columns for robustness
    ticker_col = None
    for col in results_df.columns:
        if col.lower() == 'ticker':
            ticker_col = col
            break
    if ticker_col:
        results_df = results_df[results_df[ticker_col].astype(str).str.upper().isin(valid_tickers)]
    save_results_with_versioning(results_df)
    # Add trend column if both old and new exist
    trend_map = get_bullish_score_trend_map()
    # Always add Bullish_Score_Trend column, fill with trend_map or 'Unchanged'
    results_df['ticker_clean'] = results_df['ticker'].astype(str).str.extract(r'(?:\\d+\\. )?(.*)')
    if trend_map:
        results_df['Bullish_Score_Trend'] = results_df['ticker_clean'].map(trend_map).fillna('Unchanged')
    else:
        results_df['Bullish_Score_Trend'] = 'Unchanged'
    results_df.drop('ticker_clean', axis=1, inplace=True)
    # Sort the DataFrame by Bullish_Score in descending order
    results_df = results_df.sort_values(by='Bullish_Score', ascending=False).reset_index(drop=True)

    # Update the ticker column to include the row number based on the sorted order
    results_df['ticker'] = results_df.apply(lambda row: f"{row.name + 1}. {row['ticker']}", axis=1)
    # Save the new sorted results to both the main CSV and the new versioned CSV
    results_df.to_csv(os.path.join(os.path.dirname(__file__), '../stock_csv_files/stock_analysis_comparison.csv'), index=False)
    results_df.to_csv(os.path.join(os.path.dirname(__file__), '../stock_csv_files/stock_analysis_new.csv'), index=False)
    # Clear the existing data in the Treeview
    for item in tree.get_children():
        tree.delete(item)
    # Add the new data to the Treeview
    # --- BEGIN COLOR MAPPING FOR TICKERS ---
    # Build a color palette and assign each ticker a color
    import itertools
    color_palette = [
        '#FFB347', '#77DD77', '#AEC6CF', '#FF6961', '#F49AC2', '#B39EB5', '#FFD700', '#03C03C', '#779ECB', '#C23B22',
        '#966FD6', '#FDFD96', '#CB99C9', '#DEA5A4', '#B19CD9', '#FFB347', '#B0E0E6', '#CFCFC4', '#836953', '#CFCFC4'
    ]
    # Remove row numbers from ticker for mapping
    ticker_names = results_df['ticker'].astype(str).str.replace(r'^\s*\d+\.\s*', '', regex=True).str.strip().unique()
    ticker_color_map = {ticker: color for ticker, color in zip(ticker_names, itertools.cycle(color_palette))}
    # --- END COLOR MAPPING FOR TICKERS ---

    # Define a global variable to persist the ticker color map across refreshes
    if 'persistent_ticker_color_map' not in globals():
        persistent_ticker_color_map = {}

    # Ensure the color iterator is created only once and not reset unnecessarily
    if 'color_iterator' not in globals():
        color_iterator = iter(color_palette)

    # Update the ticker_color_map with new tickers while retaining existing mappings
    new_ticker_names = results_df['ticker'].astype(str).str.replace(r'^\s*\d+\.\s*', '', regex=True).str.strip().unique()
    for ticker in new_ticker_names:
        if ticker not in persistent_ticker_color_map:
            try:
                # Assign a new color only if the ticker is not already in the map
                persistent_ticker_color_map[ticker] = next(color_iterator)
            except StopIteration:
                # If the color palette runs out, restart the iterator
                color_iterator = iter(color_palette)
                persistent_ticker_color_map[ticker] = next(color_iterator)

    # Use the persistent ticker color map
    ticker_color_map = persistent_ticker_color_map

    def get_contrast_color(hex_color):
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        # Calculate luminance
        luminance = (0.299*r + 0.587*g + 0.114*b)/255
        return '#000000' if luminance > 0.5 else '#FFFFFF'

    for index, row in results_df.iterrows():
        # Extract the clean ticker name (without row number)
        clean_ticker = str(row['ticker']).replace(f"{index+1}. ", "").strip()
        color = get_color_for_ticker(clean_ticker)
        fg_color = get_contrast_color(color)
        # Insert row with tag for color
        tree.insert("", "end", values=list(row), tags=(clean_ticker,))
        # Set both foreground and background color for the row, with contrast
        tree.tag_configure(clean_ticker, foreground=fg_color, background=color)
    # Schedule the next refresh in 60 seconds (60000 milliseconds)
    tree.after(60000, refresh_data, tree, tickers, period, interval)

def get_bullish_score_trend_map():
    """
    Returns a dict mapping ticker to 'Rising', 'Falling', or 'Unchanged' by comparing old and new CSVs.
    """
    import pandas as pd
    import os
    old_path = os.path.join(os.path.dirname(__file__), '..', 'stock_analysis_old.csv')
    new_path = os.path.join(os.path.dirname(__file__), '..', 'stock_analysis_new.csv')
    if not os.path.exists(old_path) or not os.path.exists(new_path):
        return {}
    old_df = pd.read_csv(old_path)
    new_df = pd.read_csv(new_path)
    # Remove row numbers and whitespace from ticker if present
    old_df['ticker_clean'] = old_df['ticker'].astype(str).str.replace(r'^\s*\d+\.\s*', '', regex=True).str.strip()
    new_df['ticker_clean'] = new_df['ticker'].astype(str).str.replace(r'^\s*\d+\.\s*', '', regex=True).str.strip()
    merged = pd.merge(
        old_df[['ticker_clean', 'Bullish_Score']],
        new_df[['ticker_clean', 'Bullish_Score']],
        on='ticker_clean',
        suffixes=('_old', '_new')
    )
    trend_map = {}
    for _, row in merged.iterrows():
        if row['Bullish_Score_new'] > row['Bullish_Score_old']:
            trend_map[row['ticker_clean']] = 'Rising'
        elif row['Bullish_Score_new'] < row['Bullish_Score_old']:
            trend_map[row['ticker_clean']] = 'Falling'
        else:
            trend_map[row['ticker_clean']] = 'Unchanged'
    return trend_map

def compare_bullish_scores(old_file, new_file):
    if not os.path.exists(old_file) or not os.path.exists(new_file):
        print("Comparison files not found.")
        return
    old_df = pd.read_csv(old_file)
    new_df = pd.read_csv(new_file)
    # Remove row numbers from ticker if present
    old_df['ticker'] = old_df['ticker'].astype(str).str.extract(r'(?:\\d+\\. )?(.*)')
    new_df['ticker'] = new_df['ticker'].astype(str).str.extract(r'(?:\\d+\\. )?(.*)')
    merged = pd.merge(old_df[['ticker', 'Bullish_Score']], new_df[['ticker', 'Bullish_Score']], on='ticker', suffixes=('_old', '_new'))
    up = merged[merged['Bullish_Score_new'] > merged['Bullish_Score_old']]
    down = merged[merged['Bullish_Score_new'] < merged['Bullish_Score_old']]
    if not up.empty:
        print("Stocks with increased Bullish Score:")
        print(up[['ticker', 'Bullish_Score_old', 'Bullish_Score_new']])
    else:
        print("No stocks increased in Bullish Score.")
    if not down.empty:
        print("Stocks with decreased Bullish Score:")
        print(down[['ticker', 'Bullish_Score_old', 'Bullish_Score_new']])
    else:
        print("No stocks decreased in Bullish Score.")

def save_results_with_versioning(results_df):
    old_path = os.path.join(os.path.dirname(__file__), '..', 'stock_analysis_old.csv')
    new_path = os.path.join(os.path.dirname(__file__), '..', 'stock_analysis_new.csv')
    import shutil
    if not os.path.exists(old_path):
        # First run: save to both old and new for initialization
        results_df.to_csv(old_path, index=False)
        results_df.to_csv(new_path, index=False)
        print(f"First run: results saved to {old_path} and {new_path}")
    else:
        # If new exists, rotate it to old before writing new results
        if os.path.exists(new_path):
            shutil.copyfile(new_path, old_path)
        results_df.to_csv(new_path, index=False)
        print(f"Rotated: {new_path} -> {old_path}, new results saved to {new_path}")
        compare_bullish_scores(old_path, new_path)

def main():
    # Read stock tickers from CSV file
    tickers_df = pd.read_csv('../stock_csv_files/stocks_to_analyze.csv')
    tickers = tickers_df['ticker'].tolist()

    # Prompt user for interval
    interval = input("Enter data interval (e.g., '1d' for daily, '1h' for hourly): ").lower()

    # Prompt user for period
    period = input("Enter data period (e.g., '1y' for 1 year, '6mo' for 6 months): ").lower()

    # Create the main window
    root = tk.Tk()
    root.title("Stock Analysis")

    # Configure dark grey theme
    bg_color = '#2E2E2E'
    fg_color = 'white'
    root.configure(bg=bg_color)
    root.option_add("*Background", bg_color)
    root.option_add("*Foreground", fg_color)

    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', background=bg_color, foreground=fg_color)
    style.configure('Treeview', background='#3E3E3E', foreground=fg_color, fieldbackground='#3E3E3E')
    style.configure('Treeview.Heading', background=bg_color, foreground=fg_color)
    style.map('Treeview', background=[('selected', '#4A6984')])

    # Create a frame to hold the Treeview
    frame = tk.Frame(root, bg=bg_color)
    frame.pack(expand=True, fill='both', padx=10, pady=10)

    # Create a Treeview widget
    tree = ttk.Treeview(frame)
    tree["show"] = "headings"

    # Initial data load
    refresh_data(tree, tickers, period, interval)

    # Define the column headings and set column widths after initial data load
    results_df = pd.read_csv('../stock_csv_files/stock_analysis_comparison.csv')
    # Add Bullish_Score_Trend column if present in results_df
    if 'Bullish_Score_Trend' in results_df.columns:
        columns = list(results_df.columns)
    else:
        columns = list(results_df.columns) + ['Bullish_Score_Trend']
    tree["columns"] = columns
    for col in columns:
        tree.heading(col, text=col, command=lambda _col=col: sort_treeview(tree, _col, False))
        tree.column(col, width=50)  # Set the width of each column to 50 pixels

    # Pack the Treeview widget
    tree.pack(expand=True, fill='both')

    # Bring the window to the front
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)

    # Run the application
    root.mainloop()

# Function to sort the Treeview
def sort_treeview(tree, col, reverse):
    if col in ['short_term_bullish_confidence', 'short_term_bearish_confidence', 'long_term_bullish_confidence', 'long_term_bearish_confidence', 'overall_confidence', 'latest_price', 'resistance_level', 'support_level', 'percentage_to_resistance', 'percentage_to_support', 'ATR', 'Bullish_Score']:
        l = [(float(tree.set(k, col).strip('%')), k) for k in tree.get_children('')]
    elif col == 'latest_price':
        l = [(float(tree.set(k, col).replace('$', '')), k) for k in tree.get_children('')]
    elif col == 'Bullish_Score':
        l = [(int(tree.set(k, col)), k) for k in tree.get_children('')]
    else:
        l = [(tree.set(k, col), k) for k in tree.get_children('')]

    l.sort(reverse=reverse)

    for index, (val, k) in enumerate(l):
        tree.move(k, '', index)

    tree.heading(col, command=lambda: sort_treeview(tree, col, not reverse))

if __name__ == "__main__":
    main()
