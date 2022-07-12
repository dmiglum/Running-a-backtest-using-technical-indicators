
####################################
### Loading libraries
import yfinance as yf
import ta
import pandas as pd
from datetime import date, timedelta, datetime
from IPython.display import clear_output
from itertools import product
pd.set_option('display.max_columns', 7)

### Creating helper functions
def prepare_stock_ta_backtest_data(df, start_date, end_date, strategy, **strategy_params):
  df_strategy = strategy(df, **strategy_params)
  bt_df = df_strategy[(df_strategy.index >= start_date) & (df_strategy.index <= end_date)]
  return bt_df

def get_stock_backtest_data(ticker, start, end):
    date_fmt = '%Y-%m-%d'
    start_date_buffer = datetime.strptime(start, date_fmt) - timedelta(days = 365)
    start_date_buffer = start_date_buffer.strftime(date_fmt)
    df = yf.download(ticker, start = start_date_buffer, end = end)
    
    return df

def run_stock_ta_backtest(bt_df, stop_loss_lvl=None):
  balance = 1000000
  pnl = 0
  position = 0

  last_signal = 'hold'
  last_price = 0
  c = 0

  trade_date_start = []
  trade_date_end = []
  trade_days = []
  trade_side = []
  trade_pnl = []
  trade_ret = []

  cum_value = []

  for index, row in bt_df.iterrows():
      # check and close any positions
      if row.EXIT_LONG and last_signal == 'long':
        trade_date_end.append(row.name)
        trade_days.append(c)

        pnl = (row.Open - last_price) * position
        trade_pnl.append(pnl)
        trade_ret.append((row.Open / last_price - 1) * 100)
        
        balance = balance + row.Open * position
        
        position = 0
        last_signal = 'hold'

        c = 0
      
      elif row.EXIT_SHORT and last_signal == 'short':
        trade_date_end.append(row.name)
        trade_days.append(c)
        
        pnl = (row.Open - last_price) * position
        trade_pnl.append(pnl)
        trade_ret.append((last_price / row.Open - 1) * 100)

        balance = balance + pnl

        position = 0
        last_signal = 'hold'

        c = 0


      # check signal and enter any possible position
      if row.LONG and last_signal != 'long':
        last_signal = 'long'
        last_price = row.Open
        trade_date_start.append(row.name)
        trade_side.append('long')

        position = int(balance / row.Open)
        cost = position * row.Open
        balance = balance - cost

        c = 0

      elif row.SHORT and last_signal != 'short':
        last_signal = 'short'
        last_price = row.Open
        trade_date_start.append(row.name)
        trade_side.append('short')

        position = int(balance / row.Open) * -1
        
        c = 0
      
      if stop_loss_lvl:
        # check stop loss
        if last_signal == 'long' and (row.Low / last_price - 1) * 100 <= stop_loss_lvl:
          c = c + 1

          trade_date_end.append(row.name)
          trade_days.append(c)

          stop_loss_price = last_price + round(last_price * (stop_loss_lvl / 100), 4)

          pnl = (stop_loss_price - last_price) * position
          trade_pnl.append(pnl)
          trade_ret.append((stop_loss_price / last_price - 1) * 100)
          
          balance = balance + stop_loss_price * position
          
          position = 0
          last_signal = 'hold'

          c = 0

        elif last_signal == 'short' and (last_price / row.Low - 1) * 100 <= stop_loss_lvl:
          c = c + 1

          trade_date_end.append(row.name)
          trade_days.append(c)
          
          stop_loss_price = last_price - round(last_price * (stop_loss_lvl / 100), 4)

          pnl = (stop_loss_price - last_price) * position
          trade_pnl.append(pnl)
          trade_ret.append((last_price / stop_loss_price - 1) * 100)

          balance = balance + pnl

          position = 0
          last_signal = 'hold'

          c = 0

    
      # compute market value and count days for any possible poisition
      if last_signal == 'hold':
        market_value = balance
      elif last_signal == 'long':
        c = c + 1
        market_value = position * row.Close + balance
      else: 
        c = c + 1
        market_value = (row.Close - last_price) * position + balance
      
      cum_value.append(market_value)


### Defining additional strategies
'''
Adding the following indicators:
Keltner Channel (already implemented before), Bollinger Band, Simple Moving Average, Exponential Moving Average,
MACD, RSI, Williams %R, Stochastic Fast, Stochastic Slow, Ichimoku
'''
def strategy_KeltnerChannel_origin(df, **kwargs):
  n = kwargs.get('n', 10)
  data = df.copy()

  k_band = ta.volatility.KeltnerChannel(data.High, data.Low, data.Close, n)

  data['K_BAND_UB'] = k_band.keltner_channel_hband().round(4)
  data['K_BAND_LB'] = k_band.keltner_channel_lband().round(4)

  data['CLOSE_PREV'] = data.Close.shift(1)
  
  data['LONG'] = (data.Close <= data.K_BAND_LB) & (data.CLOSE_PREV > data.K_BAND_LB)
  data['EXIT_LONG'] = (data.Close >= data.K_BAND_UB) & (data.CLOSE_PREV < data.K_BAND_UB)

  data['SHORT'] = (data.Close >= data.K_BAND_UB) & (data.CLOSE_PREV < data.K_BAND_UB)
  data['EXIT_SHORT'] = (data.Close <= data.K_BAND_LB) & (data.CLOSE_PREV > data.K_BAND_LB)

  data.LONG = data.LONG.shift(1)
  data.EXIT_LONG = data.EXIT_LONG.shift(1)
  data.SHORT = data.SHORT.shift(1)
  data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

  return data

def strategy_BollingerBands(df, **kwargs):
  n = kwargs.get('n', 10)
  n_rng = kwargs.get('n_rng', 2)
  data = df.copy()

  boll = ta.volatility.BollingerBands(data.Close, n, n_rng)

  data['BOLL_LBAND_INDI'] = boll.bollinger_lband_indicator()
  data['BOLL_UBAND_INDI'] = boll.bollinger_hband_indicator()

  data['CLOSE_PREV'] = data.Close.shift(1)

  data['LONG'] = data.BOLL_LBAND_INDI == 1
  data['EXIT_LONG'] = data.BOLL_UBAND_INDI == 1

  data['SHORT'] = data.BOLL_UBAND_INDI == 1
  data['EXIT_SHORT'] = data.BOLL_LBAND_INDI == 1

  data.LONG = data.LONG.shift(1)
  data.EXIT_LONG = data.EXIT_LONG.shift(1)
  data.SHORT = data.SHORT.shift(1)
  data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

  return data
    
# moving average takes a ma parameter to compute either sma or ema
def strategy_MA(df, **kwargs):
  n = kwargs.get('n', 50)
  ma_type = kwargs.get('ma_type', 'sma')
  ma_type = ma_type.strip().lower()
  data = df.copy()
  
  if ma_type == 'sma':
    sma = ta.trend.SMAIndicator(data.Close, n)
    data['MA'] = sma.sma_indicator().round(4)
  elif ma_type == 'ema':
    ema = ta.trend.EMAIndicator(data.Close, n)
    data['MA'] = ema.ema_indicator().round(4)

  data['CLOSE_PREV'] = data.Close.shift(1)

  data['LONG'] = (data.Close > data.MA) & (data.CLOSE_PREV <= data.MA)
  data['EXIT_LONG'] = (data.Close < data.MA) & (data.CLOSE_PREV >= data.MA)

  data['SHORT'] = (data.Close < data.MA) & (data.CLOSE_PREV >= data.MA)
  data['EXIT_SHORT'] = (data.Close > data.MA) & (data.CLOSE_PREV <= data.MA)

  data.LONG = data.LONG.shift(1)
  data.EXIT_LONG = data.EXIT_LONG.shift(1)
  data.SHORT = data.SHORT.shift(1)
  data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

  return data

def strategy_MACD(df, **kwargs):
  n_slow = kwargs.get('n_slow', 26)
  n_fast = kwargs.get('n_fast', 12)
  n_sign = kwargs.get('n_sign', 9)
  data = df.copy()

  macd = ta.trend.MACD(data.Close, n_slow, n_fast, n_sign)

  data['MACD_DIFF'] = macd.macd_diff().round(4)
  data['MACD_DIFF_PREV'] = data.MACD_DIFF.shift(1)

  data['LONG'] = (data.MACD_DIFF > 0) & (data.MACD_DIFF_PREV <= 0)
  data['EXIT_LONG'] = (data.MACD_DIFF < 0) & (data.MACD_DIFF_PREV >= 0)

  data['SHORT'] = (data.MACD_DIFF < 0) & (data.MACD_DIFF_PREV >= 0)
  data['EXIT_SHORT'] = (data.MACD_DIFF > 0) & (data.MACD_DIFF_PREV <= 0)

  data.LONG = data.LONG.shift(1)
  data.EXIT_LONG = data.EXIT_LONG.shift(1)
  data.SHORT = data.SHORT.shift(1)
  data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

  return data

def strategy_RSI(df, **kwargs):
  n = kwargs.get('n', 14)
  data = df.copy()

  rsi = ta.momentum.RSIIndicator(data.Close, n)

  data['RSI'] = rsi.rsi().round(4)
  data['RSI_PREV'] = data.RSI.shift(1)

  data['LONG'] = (data.RSI > 30) & (data.RSI_PREV <= 30)
  data['EXIT_LONG'] = (data.RSI < 70) & (data.RSI_PREV >= 70)

  data['SHORT'] = (data.RSI < 70) & (data.RSI_PREV >= 70)
  data['EXIT_SHORT'] = (data.RSI > 30) & (data.RSI_PREV <= 30)

  data.LONG = data.LONG.shift(1)
  data.EXIT_LONG = data.EXIT_LONG.shift(1)
  data.SHORT = data.SHORT.shift(1)
  data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

  return data

def strategy_WR(df, **kwargs):
  n = kwargs.get('n', 14)
  data = df.copy()

  wr = ta.momentum.WilliamsRIndicator(data.High, data.Low, data.Close, n)

  data['WR'] = wr.williams_r().round(4)
  data['WR_PREV'] = data.WR.shift(1)

  data['LONG'] = (data.WR > -80) & (data.WR_PREV <= -80)
  data['EXIT_LONG'] = (data.WR < -20) & (data.WR_PREV >= -20)

  data['SHORT'] = (data.WR < -20) & (data.WR_PREV >= -20)
  data['EXIT_SHORT'] = (data.WR > -80) & (data.WR_PREV <= -80)

  data.LONG = data.LONG.shift(1)
  data.EXIT_LONG = data.EXIT_LONG.shift(1)
  data.SHORT = data.SHORT.shift(1)
  data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

  return data

def strategy_Stochastic_fast(df, **kwargs):
  k = kwargs.get('k', 20)
  d = kwargs.get('d', 5)
  data = df.copy()

  sto = ta.momentum.StochasticOscillator(data.High, data.Low, data.Close, k, d)

  data['K'] = sto.stoch().round(4)
  data['D'] = sto.stoch_signal().round(4)
  data['DIFF'] = data['K'] - data['D']
  data['DIFF_PREV'] = data.DIFF.shift(1)
  
  data['LONG'] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)
  data['EXIT_LONG'] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)

  data['SHORT'] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)
  data['EXIT_SHORT'] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)

  data.LONG = data.LONG.shift(1)
  data.EXIT_LONG = data.EXIT_LONG.shift(1)

  data.SHORT = data.SHORT.shift(1)
  data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

  return data

def strategy_Stochastic_slow(df, **kwargs):
  k = kwargs.get('k', 20)
  d = kwargs.get('d', 5)
  dd = kwargs.get('dd', 3)
  data = df.copy()

  sto = ta.momentum.StochasticOscillator(data.High, data.Low, data.Close, k, d)

  data['K'] = sto.stoch().round(4)
  data['D'] = sto.stoch_signal().round(4)
  
  ma = ta.trend.SMAIndicator(data.D, dd)
  data['DD'] = ma.sma_indicator().round(4)

  data['DIFF'] = data['D'] - data['DD']
  data['DIFF_PREV'] = data.DIFF.shift(1)
  
  data['LONG'] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)
  data['EXIT_LONG'] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)

  data['SHORT'] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)
  data['EXIT_SHORT'] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)

  data.LONG = data.LONG.shift(1)
  data.EXIT_LONG = data.EXIT_LONG.shift(1)

  data.SHORT = data.SHORT.shift(1)
  data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

  return data  
      
def strategy_Ichmoku(df, **kwargs):
  n_conv = kwargs.get('n_conv', 9)
  n_base = kwargs.get('n_base', 26)
  n_span_b = kwargs.get('n_span_b', 26)
  data = df.copy()

  ichmoku = ta.trend.IchimokuIndicator(data.High, data.Low, n_conv, n_base, n_span_b)

  data['BASE'] = ichmoku.ichimoku_base_line().round(4)
  data['CONV'] = ichmoku.ichimoku_conversion_line().round(4)

  data['DIFF'] = data['CONV'] - data['BASE']
  data['DIFF_PREV'] = data.DIFF.shift(1)
  
  data['LONG'] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)
  data['EXIT_LONG'] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)

  data['SHORT'] = (data.DIFF < 0) & (data.DIFF_PREV >= 0)
  data['EXIT_SHORT'] = (data.DIFF > 0) & (data.DIFF_PREV <= 0)

  data.LONG = data.LONG.shift(1)
  data.EXIT_LONG = data.EXIT_LONG.shift(1)

  data.SHORT = data.SHORT.shift(1)
  data.EXIT_SHORT = data.EXIT_SHORT.shift(1)

  return data    
    

### Putting strategies in a good structure
'''
With strategies defined, we need to put them in a good data structure so that we can loop them through for backtesting with possible 
combinations of inputs. First, our data structure should be a list, because we have a list of strategies to loop through. 
Second, we need to define all possible inputs with the parameter names, so this sounds like a dictionary. Finally, 
when we loop through the strategies, we need to have a generic name to get the strategy function as well 
as possible parameters, so each element in the list should be in a dictionary too. Ultimately, our data structure will be a list of dictionaries.
'''
strategies = [
  {
    'func': strategy_KeltnerChannel_origin,
    'param': {
      'n': [i for i in range(10, 35, 5)]
    }
  },

  {
    'func': strategy_BollingerBands,
    'param': {
      'n': [i for i in range(10, 35, 5)],
      'n_rng': [1, 2]
    }
  },

  {
    'func': strategy_MA,
    'param': {
      'n': [i for i in range(10, 110, 10)],
      'ma_type': ['sma', 'ema']
    }
  },

  {
    'func': strategy_MACD,
    'param': {
      'n_slow': [i for i in range(10, 16)],
      'n_fast': [i for i in range(20, 26)],
      'n_sign': [i for i in range(5, 11)]
    }
  },

  {
    'func': strategy_RSI,
    'param': {
      'n': [i for i in range(5, 21)]
    }
  },

  {
    'func': strategy_WR,
    'param': {
      'n': [i for i in range(5, 21)]
    }
  },

  {
    'func': strategy_Stochastic_fast,
    'param': {
      'k': [i for i in range(15, 26)],
      'd': [i for i in range(5, 11)]
    }
  },

  {
    'func': strategy_Stochastic_slow,
    'param': {
      'k': [i for i in range(15, 26)],
      'd': [i for i in range(5, 11)],
      'dd': [i for i in range(1, 6)]
    }
  },

  {
    'func': strategy_Ichmoku,
    'param': {
      'n_conv': [i for i in range(5, 16)],
      'n_base': [i for i in range(20, 36)],
      'n_span_b': [26]
    }
  },
]

'''
We defined a variable “strategies”, which is a list. Within this list, we have dictionaries defining what we need to test with. 
The key “func” will get us the strategy function, and the key “param” will get us a dictionary of all parameter names with the possible inputs as lists.
'''

### Looping through the strategies and generating unique combinations
'''
For each loop, we will get the strategy with the key “func” and parameter dictionary with the key “param”. 
Because we don’t know how many parameters we will have, we have another loop to get the parameter name and its 
possible inputs as a list. We then put parameter names and possible inputs into the list “param_name” and “param_list”. 
Finally, we turn a list of unique combinations to a list of dictionaries.
'''

### Running a backtest
ticker = 'NVDA'
start_date = '2010-01-01'
end_date = '2019-12-31'

df = get_stock_backtest_data(ticker, start_date, end_date)

stop_loss_lvl = [-i for i in range(2, 6, 1)]
stop_loss_lvl.append(None)

result_dict = {
    'strategy': [],
    'param': [],
    'stoploss': [],
    'return': [],
    'max_drawdown': []
}



for s in strategies:
  func = s['func']
  param = s['param']

  strategy_name = str(func).split(' ')[1]

  param_name = []
  param_list = []

  for k in param:
    param_name.append(k)
    param_list.append(param[k])

  param_dict_list = [dict(zip(param_name, param)) for param in list(product(*param_list))]
  total_param_dict = len(param_dict_list)

  c = 0

  for param_dict in param_dict_list:
    clear_output()
    c = c + 1
    print('Running backtest for {} - ({}/{})'.format(strategy_name, c, total_param_dict))

    for l in stop_loss_lvl:
      bt_df = prepare_stock_ta_backtest_data(
          df, start_date, end_date, 
          func, **param_dict)

      result = run_stock_ta_backtest(bt_df, stop_loss_lvl=l)

      result_dict['strategy'].append(strategy_name)
      result_dict['param'].append(str(param_dict))
      result_dict['stoploss'].append(l)
      result_dict['return'].append(result['cum_ret_df'].iloc[-1, 0])
      result_dict['max_drawdown'].append(result['max_drawdown']['pct'])


df = pd.DataFrame(result_dict)
df2 = df.sort_values('return', ascending=False)

