import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MultipleLocator
from IPython.display import display
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# stats for entire data set
def summary_stats(fund_list, df, period):
    if period == 'Monthly':
        timeframe = 12 # months
        df_stats = pd.DataFrame(df.mean(axis=0) * timeframe) # annualized
        # df_stats = pd.DataFrame((1 + df.mean(axis=0)) ** timeframe - 1) # annualized, this is this true annualized return but we will simply use the mean
        df_stats.columns = ['Annualized Mean']
        df_stats['Annualized Volatility'] = df.std() * np.sqrt(timeframe) # annualized
        df_stats['Annualized Sharpe Ratio'] = df_stats['Annualized Mean'] / df_stats['Annualized Volatility']

        df_cagr = (1 + df['Return']).cumprod()
        cagr = (df_cagr[-1] / 1) ** (1/(len(df_cagr) / timeframe)) - 1
        df_stats['CAGR'] = cagr

        df_stats[period + ' Max Return'] = df.max()
        df_stats[period + ' Max Return (Date)'] = df.idxmax().values[0]
        df_stats[period + ' Min Return'] = df.min()
        df_stats[period + ' Min Return (Date)'] = df.idxmin().values[0]
        
        wealth_index = 1000*(1+df).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks)/previous_peaks

        df_stats['Max Drawdown'] = drawdowns.min()
        df_stats['Peak'] = [previous_peaks[col][:drawdowns[col].idxmin()].idxmax() for col in previous_peaks.columns]
        df_stats['Bottom'] = drawdowns.idxmin()
    
        recovery_date = []
        for col in wealth_index.columns:
            prev_max = previous_peaks[col][:drawdowns[col].idxmin()].max()
            recovery_wealth = pd.DataFrame([wealth_index[col][drawdowns[col].idxmin():]]).T
            recovery_date.append(recovery_wealth[recovery_wealth[col] >= prev_max].index.min())
        df_stats['Recovery Date'] = recovery_date

        plan_name = '_'.join(fund_list)
        file = plan_name + "_Summary_Stats.xlsx"
        # location = "Crypto_Trend_Following/" + file
        location = file
        # df_stats.to_excel(location, sheet_name='data')
        print(f"Summary stats complete for {plan_name}.")
        return df_stats
    
    elif period == 'Weekly':
        timeframe = 52 # weeks
        df_stats = pd.DataFrame(df.mean(axis=0) * timeframe) # annualized
        # df_stats = pd.DataFrame((1 + df.mean(axis=0)) ** timeframe - 1) # annualized, this is this true annualized return but we will simply use the mean
        df_stats.columns = ['Annualized Mean']
        df_stats['Annualized Volatility'] = df.std() * np.sqrt(timeframe) # annualized
        df_stats['Annualized Sharpe Ratio'] = df_stats['Annualized Mean'] / df_stats['Annualized Volatility']

        df_cagr = (1 + df['Return']).cumprod()
        cagr = (df_cagr[-1] / 1) ** (1/(len(df_cagr) / timeframe)) - 1
        df_stats['CAGR'] = cagr

        df_stats[period + ' Max Return'] = df.max()
        df_stats[period + ' Max Return (Date)'] = df.idxmax().values[0]
        df_stats[period + ' Min Return'] = df.min()
        df_stats[period + ' Min Return (Date)'] = df.idxmin().values[0]
        
        wealth_index = 1000*(1+df).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks)/previous_peaks

        df_stats['Max Drawdown'] = drawdowns.min()
        df_stats['Peak'] = [previous_peaks[col][:drawdowns[col].idxmin()].idxmax() for col in previous_peaks.columns]
        df_stats['Bottom'] = drawdowns.idxmin()
    
        recovery_date = []
        for col in wealth_index.columns:
            prev_max = previous_peaks[col][:drawdowns[col].idxmin()].max()
            recovery_wealth = pd.DataFrame([wealth_index[col][drawdowns[col].idxmin():]]).T
            recovery_date.append(recovery_wealth[recovery_wealth[col] >= prev_max].index.min())
        df_stats['Recovery Date'] = recovery_date

        plan_name = '_'.join(fund_list)
        file = plan_name + "_Summary_Stats.xlsx"
        # location = "Crypto_Trend_Following/" + file
        location = file
        # df_stats.to_excel(location, sheet_name='data')
        print(f"Summary stats complete for {plan_name}.")
        return df_stats
        
    elif period == 'Daily':
        timeframe = 365 # days
        df_stats = pd.DataFrame(df.mean(axis=0) * timeframe) # annualized
        # df_stats = pd.DataFrame((1 + df.mean(axis=0)) ** timeframe - 1) # annualized, this is this true annualized return but we will simply use the mean
        df_stats.columns = ['Annualized Mean']
        df_stats['Annualized Volatility'] = df.std() * np.sqrt(timeframe) # annualized
        df_stats['Annualized Sharpe Ratio'] = df_stats['Annualized Mean'] / df_stats['Annualized Volatility']

        df_cagr = (1 + df['Return']).cumprod()
        cagr = (df_cagr[-1] / 1) ** (1/(len(df_cagr) / timeframe)) - 1
        df_stats['CAGR'] = cagr
        
        df_stats[period + ' Max Return'] = df.max()
        df_stats[period + ' Max Return (Date)'] = df.idxmax().values[0]
        df_stats[period + ' Min Return'] = df.min()
        df_stats[period + ' Min Return (Date)'] = df.idxmin().values[0]
        
        wealth_index = 1000*(1+df).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks)/previous_peaks

        df_stats['Max Drawdown'] = drawdowns.min()
        df_stats['Peak'] = [previous_peaks[col][:drawdowns[col].idxmin()].idxmax() for col in previous_peaks.columns]
        df_stats['Bottom'] = drawdowns.idxmin()
    
        recovery_date = []
        for col in wealth_index.columns:
            prev_max = previous_peaks[col][:drawdowns[col].idxmin()].max()
            recovery_wealth = pd.DataFrame([wealth_index[col][drawdowns[col].idxmin():]]).T
            recovery_date.append(recovery_wealth[recovery_wealth[col] >= prev_max].index.min())
        df_stats['Recovery Date'] = recovery_date

        plan_name = '_'.join(fund_list)
        file = plan_name + "_Summary_Stats.xlsx"
        # location = "Crypto_Trend_Following/" + file
        location = file
        # df_stats.to_excel(location, sheet_name='data')
        print(f"Summary stats complete for {plan_name}.")
        return df_stats
            
    else:
        return print("Error, check inputs")