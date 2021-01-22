import pandas as pd
import numpy as np

#Technical-analysis
from ta.trend import ADXIndicator, AroonIndicator, CCIIndicator
from ta.momentum import StochRSIIndicator, AwesomeOscillatorIndicator, StochasticOscillator, WilliamsRIndicator

from ta.volume import OnBalanceVolumeIndicator


'''
สร้างอินดิเคเตอร์
trend แนวโน้มราคา
Momentum แรงเหวี่ยงของราคา
Votatility ความผันผวนของราคา


'''

class Indicator():

  def MA(self,df,n):
    df['MA'] = df['Close'].rolling(n).mean() 
    return df
  
  def MACD(self,df,n1,n2,n3,no):#trend 
    '''
    MACD range(-inf,inf)
    '''
    df_MACD = df.copy()
    # df_MACD['EWMA-5'] = df_MACD['Close'].ewm(span=5).mean()
    df_MACD['EWMA-12'] = df_MACD['Close'].ewm(span=n1).mean()
    df_MACD['EWMA-26'] = df_MACD['Close'].ewm(span=n2).mean()
    df['MACD-'+str(no)] = df_MACD['EWMA-12'] - df_MACD['EWMA-26']
    df['SIGNAL LINE'+str(no)] = df['MACD-'+str(no)].ewm(span=n3).mean() 

    # df['MACD'] = np.where((df_MACD['MACD']>df_MACD['SIGNAL LINE'])&(df_MACD['MACD'].shift()<df_MACD['SIGNAL LINE'].shift()),'buy',
    #                           (np.where((df_MACD['MACD'] < df_MACD['SIGNAL LINE'])  & (df_MACD['MACD'].shift()>df_MACD['SIGNAL LINE'].shift()),'sell','wait or hold')))

    return df

  def DMI(self,df,n):#trend
    df_DMI = df.copy()
    adxi = ADXIndicator(df['High'],df['Low'],df['Close'],n,False)
    df['plusDI'] = adxi.adx_pos()
    df['minusDI'] = adxi.adx_neg()
    df['adx'] = adxi.adx()

    # df['DMI'] = np.where((df_DMI['+DI']>df_DMI['-DI'])&(df_DMI['+DI'].shift()<df_DMI['-DI'].shift()),'buy',
    #                           (np.where((df_DMI['+DI'] < df_DMI['-DI'])  & (df_DMI['+DI'].shift()>df_DMI['-DI'].shift()),'sell','wait or hold')))
    
    return df
  
  def STOCHRSI(self,df):#momentum
    '''
    data range (0,100)
    '''
    df_STRSI = df.copy()
    strsi = StochRSIIndicator(df['Close'])
    df['storsi'] = strsi.stochrsi()
    df['%K'] = strsi.stochrsi_k()
    df['%D'] = df['%K'].ewm(span=3).mean()

    # df['STOCHRSI']  = np.where((df_STRSI['%K']>df_STRSI['%D'])&(df_STRSI['%K'].shift()<df_STRSI['%D'].shift())&(df_STRSI['%K']<20),'buy',
    #                             (np.where((df_STRSI['%K'] < df_STRSI['%D']) & (df_STRSI['%K'].shift()>df_STRSI['%D'].shift()) & (df_STRSI['%K']>80),'sell','wait or hold')))

    return df

  def RSI(self,df):
    def computeRSI (data, time_window):
      diff = data.diff(1).dropna()        
      up_chg = 0 * diff
      down_chg = 0 * diff
      up_chg[diff > 0] = diff[ diff>0 ]
      down_chg[diff < 0] = diff[ diff < 0 ]
      up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
      down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
      rs = abs(up_chg_avg/down_chg_avg)
      rsi = 100 - 100/(1+rs)
      return rsi
    df['RSI-7'] = computeRSI(df['Close'],7)
    df['RSI-14'] = computeRSI(df['Close'], 14)
    df['RSI'] = computeRSI(df['Close'],14)
    return df

  def STOCHASTIC(self,df):
    df_STOCH = df.copy()
    stoch = StochasticOscillator(df['High'],df['Low'],df['Close'])
    df['%K'] = stoch.stoch()
    df['%D'] = stoch.stoch_signal()
    return df

  def WILLIANSR(self,df):
    williamsr = WilliamsRIndicator(df['High'],df['Low'],df['Close'])
    df['%R'] = williamsr.williams_r()
    return df


  def AROON(self,df):#trend
    '''
    data range (0%,100%)
    '''
    df_AROON = df.copy()
    aroon = AroonIndicator(df['Close'])
    df['aroon_up'] = aroon.aroon_up()
    df['aroon_down'] = aroon.aroon_down()

    return df

  def ATR(self,df):#Volatility
    '''
    data range(0,inf)
    '''
    df_ATR = df.copy()
    df_ATR['H(t) - L(t)'] = df['High'] - df['Low']
    df_ATR['H(t) - C(Y)'] = df['High'] - df['Close'].shift(1)
    df_ATR['L(t) - C(Y)'] = df['Low'] - df['Close'].shift(1)
    selected_columns = df_ATR[['H(t) - L(t)','H(t) - C(Y)','L(t) - C(Y)']]
    df['ATR'] = selected_columns.max(axis = 1)

    return df
  
  def CCI(self,df):#trend
    '''
    data range(-inf,inf)
    '''
    df_CCI = df.copy()
    cci = CCIIndicator(df['High'],df['Low'],df['Close'])
    df['CCI'] = cci.cci()
    return df

  def OBV(self,df):
    '''
    data range(0,inf)
    '''
    df_OBV = df.copy()
    obv = OnBalanceVolumeIndicator(df['Close'],df['Volume'])
    df['OBV'] = obv.on_balance_volume()
    return df

  def AO(self,df):
    '''
    data range(-inf,inf)
    '''
    df_AO = df.copy()
    AweOs = AwesomeOscillatorIndicator(df['High'],df['Low'])
    df['AO'] = AweOs.awesome_oscillator()
    return df

  def check_trend(self,df):
    df.loc[(df['Close'] > df['MA']) &
    ((df['Close'] > df['Close'].shift(1)) | (df['MA'] > df['MA'].shift(1))) &
    ((df['Close'] > df['Close'].shift(2)) | (df['MA'] > df['MA'].shift(2))) &
    ((df['Close'] > df['Close'].shift(3)) | (df['MA'] > df['MA'].shift(3))) &
    ((df['Close'] > df['Close'].shift(4)) | (df['MA'] > df['MA'].shift(4))) &
    ((df['Close'] > df['Close'].shift(5)) | (df['MA'] > df['MA'].shift(5))) , 'Trend'] = 'Up'

    df.loc[(df['Close'] < df['MA']) &
    ((df['Close'] < df['Close'].shift(1)) | (df['MA'] < df['MA'].shift(1))) &
    ((df['Close'] < df['Close'].shift(2)) | (df['MA'] < df['MA'].shift(2))) &
    ((df['Close'] < df['Close'].shift(3)) | (df['MA'] < df['MA'].shift(3))) &
    ((df['Close'] < df['Close'].shift(4)) | (df['MA'] < df['MA'].shift(4))) &
    ((df['Close'] < df['Close'].shift(5)) | (df['MA'] < df['MA'].shift(5))) , 'Trend'] = 'Down'
    
    df.loc[df['Trend'].isnull(), 'Ntrend'] = df['Trend'].shift(1)
    df.loc[df['Trend'].isnull() & df['Trend'].shift(1).isnull(),'Ntrend'] = df['Trend'].shift(2)
    df.loc[df['Trend'].isnull() & df['Trend'].shift(1).isnull() & df['Trend'].shift(2).isnull(),'Ntrend'] = df['Trend'].shift(3)
    df.loc[df['Trend'].isnull() & df['Trend'].shift(1).isnull() & df['Trend'].shift(2).isnull() & 
    df['Trend'].shift(3).isnull(),'Ntrend'] = df['Trend'].shift(4)
    df.loc[df['Trend'].isnull() & df['Trend'].shift(1).isnull() & df['Trend'].shift(2).isnull() & 
    df['Trend'].shift(3).isnull() & df['Trend'].shift(4).isnull(),'Ntrend'] = df['Trend'].shift(5)
    df.loc[df['Trend'].isnull() & df['Trend'].shift(1).isnull() & df['Trend'].shift(2).isnull() & 
    df['Trend'].shift(3).isnull() & df['Trend'].shift(4).isnull() & df['Trend'].shift(5).isnull(),'Ntrend'] = df['Trend'].shift(6)
    
    df['S1'] = df['Close'].shift(-1)
    df['S2'] = df['Close'].shift(-2)
    df.loc[df['S1'].isnull(),'S1'] = 0
    df.loc[df['S2'].isnull(),'S2'] = 0

   
    #Trading Signal (0,1)
    df.loc[(df['Trend'] == 'Up') | (df['Ntrend'] == 'Up'), 'Trading Signal'] = ((abs(df['Close'] - df[['Close','S1','S2']].min(axis = 1)) / 
    abs(df[['Close','S1','S2']].max(axis = 1) - df[['Close','S1','S2']].min(axis = 1)))*0.5) + 0.5
    df.loc[(df['Trend'] == 'Down') | (df['Ntrend'] == 'Down'), 'Trading Signal'] = ((abs(df['Close'] - df[['Close','S1','S2']].min(axis = 1)) / 
    abs(df[['Close','S1','S2']].max(axis = 1) - df[['Close','S1','S2']].min(axis = 1)))*0.5)
   
    df.loc[df['Trend'].isnull(),'Trend'] =df['Ntrend']
    
    df = df.drop(['Ntrend','S1','S2'],axis=1)
        
    return df

