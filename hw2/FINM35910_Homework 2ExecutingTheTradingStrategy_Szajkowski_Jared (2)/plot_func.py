import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MultipleLocator
from IPython.display import display
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x: '%.4f' % x)

def plot_cumulative_return(strat_df, fund_list, initial_investment=10000):
    # Create dataframe for prices
    buy_hold = pd.DataFrame(index = strat_df.index)

    # Define color palette for each fund
    colors = list(mcolors.TABLEAU_COLORS.values())

    for fund in fund_list:
        col_name = fund + "_Close"
        buy_hold = buy_hold.join(strat_df[[col_name]])

    # Calculate shares & value
    for fund in fund_list:
        buy_hold[fund + "_Shares"] = initial_investment / buy_hold.iloc[0][fund + "_Close"]
        buy_hold[fund + "_Value"] = buy_hold[fund + "_Shares"] * buy_hold[fund + "_Close"]
        buy_hold[fund + "_Return"] = buy_hold[fund + "_Value"].pct_change()
        buy_hold[fund + "_Cumulative_Return"] = (1 + buy_hold[fund + "_Return"]).cumprod()

    # Generate plot   
    plt.figure(figsize=(10, 5), facecolor = "#F5F5F5")
    
    # Plotting data
    plt.plot(strat_df.index, strat_df['Cumulative_Return'], label = "Trend Following Model Cumulative Return", linestyle = "-", color = "green", linewidth = 1)
    for idx, fund in enumerate(fund_list):
        plt.plot(buy_hold.index, buy_hold[fund + "_Cumulative_Return"], label = fund + " Buy & Hold Cumulative Return", linestyle = "-", color = colors[idx % len(colors)], linewidth = 1)
    
    # Set X axis
    # x_tick_spacing = 5  # Specify the interval for x-axis ticks
    # plt.gca().xaxis.set_major_locator(MultipleLocator(x_tick_spacing))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xlabel("Year", fontsize = 9)
    plt.xticks(rotation = 0, fontsize = 7)

    # Set Y axis
    max_value = strat_df['Cumulative_Return'].max()
    if max_value >= 1500000:
        round_value = -5
    elif max_value >= 150000:
        round_value = -4    
    elif max_value >= 15000:
        round_value = -3
    elif max_value >= 1500:
        round_value = -2
    else:
        round_value = -1
    y_tick_spacing = round(max_value / 15, round_value) # Specify the interval for y-axis ticks
    plt.gca().yaxis.set_major_locator(MultipleLocator(y_tick_spacing))
    plt.ylabel("Cumulative Return", fontsize = 9)
    plt.yticks(fontsize = 7)

    # Set title, etc.
    plt.title("Cumulative Return For Trend Following Model vs Individual Asset Buy & Hold", fontsize = 12)
    
    # Set the grid & legend
    plt.tight_layout()
    plt.grid(True)
    plt.legend(fontsize=7)

    # Save the figure
    # plt.savefig('Crypto_Trend_Following/Portfolio_Cumulative_Return.png', dpi=300, bbox_inches='tight')
    # plt.savefig("Portfolio_Cumulative_Return.png", dpi = 300, bbox_inches = "tight")

    # Display the plot
    return plt.show()

def plot_values(strat_df):   
    # Generate plot   
    plt.figure(figsize=(10, 5), facecolor = "#F5F5F5")
    
    # Plotting data
    plt.plot(strat_df.index, strat_df['Total_AA_$_Invested'], label = "Equity Position Value", linestyle = "-", color = "blue", linewidth = 1)
    plt.plot(strat_df.index, strat_df['Cash_AA'], label = "Cash Position Value", linestyle = "-", color = "green", linewidth = 1)
    plt.plot(strat_df.index, strat_df['Total_Value'], label = "Total Portfolio Value", linestyle = "-", color = "black", linewidth = 1)
    
    # Set X axis
    # x_tick_spacing = 5  # Specify the interval for x-axis ticks
    # plt.gca().xaxis.set_major_locator(MultipleLocator(x_tick_spacing))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xlabel("Year", fontsize = 9)
    plt.xticks(rotation = 0, fontsize = 7)

    # Set Y axis
    max_value = strat_df['Total_Value'].max()
    if max_value >= 1500000:
        round_value = -5
    elif max_value >= 150000:
        round_value = -4    
    elif max_value >= 15000:
        round_value = -3
    elif max_value >= 1500:
        round_value = -2
    else:
        round_value = -1
    y_tick_spacing = round(max_value / 15, round_value) # Specify the interval for y-axis ticks
    plt.gca().yaxis.set_major_locator(MultipleLocator(y_tick_spacing))
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))) # Adding commas to y-axis labels
    plt.ylabel("Total Value ($)", fontsize = 9)
    plt.yticks(fontsize = 7)

    # Set title, etc.
    plt.title("Trend Following Model Equity Position Value, Cash Position Value, and Total Portfolio Value", fontsize = 12)
    
    # Set the grid & legend
    plt.tight_layout()
    plt.grid(True)
    plt.legend(fontsize=7)

    # Save the figure
    # plt.savefig('Crypto_Trend_Following/Portfolio_Values.png', dpi=300, bbox_inches='tight')
    # plt.savefig("Portfolio_Values.png", dpi = 300, bbox_inches = "tight")

    # Display the plot
    return plt.show()

def plot_buy_hold(strat_df, fund_list, initial_investment=10000):
    # Create dataframe for prices
    buy_hold = pd.DataFrame(index = strat_df.index)

    # Define color palette for each fund
    colors = list(mcolors.TABLEAU_COLORS.values())

    for fund in fund_list:
        col_name = fund + "_Close"
        buy_hold = buy_hold.join(strat_df[[col_name]])

    # Calculate shares & value
    for fund in fund_list:
        buy_hold[fund + "_Shares"] = initial_investment / buy_hold.iloc[0][fund + "_Close"]
        buy_hold[fund + "_Value"] = buy_hold[fund + "_Shares"] * buy_hold[fund + "_Close"]
        buy_hold[fund + "_Return"] = buy_hold[fund + "_Value"].pct_change()
        buy_hold[fund + "_Cumulative_Return"] = (1 + buy_hold[fund + "_Return"]).cumprod()

    # Generate plot   
    plt.figure(figsize=(10, 5), facecolor = "#F5F5F5")
    
    # Plotting data
    plt.plot(strat_df.index, strat_df['Total_Value'], label = "Trend Following Model Total Portfolio Value", linestyle = "-", color = "black", linewidth = 1)
    for idx, fund in enumerate(fund_list):
        plt.plot(buy_hold.index, buy_hold[fund + "_Value"], label = fund + " Buy & Hold Value", linestyle = "-", color = colors[idx % len(colors)], linewidth = 1)
    
    # Set X axis
    # x_tick_spacing = 5  # Specify the interval for x-axis ticks
    # plt.gca().xaxis.set_major_locator(MultipleLocator(x_tick_spacing))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xlabel("Year", fontsize = 9)
    plt.xticks(rotation = 0, fontsize = 7)

    # Set Y axis
    max_value = strat_df['Total_Value'].max()
    if max_value >= 1500000:
        round_value = -5
    elif max_value >= 150000:
        round_value = -4    
    elif max_value >= 15000:
        round_value = -3
    elif max_value >= 1500:
        round_value = -2
    else:
        round_value = -1
    y_tick_spacing = round(max_value / 15, round_value) # Specify the interval for y-axis ticks
    plt.gca().yaxis.set_major_locator(MultipleLocator(y_tick_spacing))
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))) # Adding commas to y-axis labels
    plt.ylabel("Total Value ($)", fontsize = 9)
    plt.yticks(fontsize = 7)

    # Set title, etc.
    plt.title("Portfolio Values For Trend Following Model vs Individual Asset Buy & Hold W/ $10,000 Initial Allocation", fontsize = 12)
    
    # Set the grid & legend
    plt.tight_layout()
    plt.grid(True)
    plt.legend(fontsize=7)

    # Save the figure
    # plt.savefig('Crypto_Trend_Following/Portfolio_Value_Comparison.png', dpi=300, bbox_inches='tight')
    # plt.savefig("Portfolio_Value_Comparison.png", dpi = 300, bbox_inches = "tight")

    # Display the plot
    return plt.show()

def plot_drawdown(strat_df, fund_list, initial_investment=10000):
    # Create dataframe for prices
    buy_hold = pd.DataFrame(index = strat_df.index)

    # Define color palette for each fund
    colors = list(mcolors.TABLEAU_COLORS.values())

    for fund in fund_list:
        col_name = fund + "_Close"
        buy_hold = buy_hold.join(strat_df[[col_name]])

    # Calculate shares & value
    for fund in fund_list:
        buy_hold[fund + "_Shares"] = initial_investment / buy_hold.iloc[0][fund + "_Close"]
        buy_hold[fund + "_Value"] = buy_hold[fund + "_Shares"] * buy_hold[fund + "_Close"]
        buy_hold[fund + "_Return"] = buy_hold[fund + "_Value"].pct_change()
        buy_hold[fund + "_Cumulative_Return"] = (1 + buy_hold[fund + "_Return"]).cumprod()


    # Calculate rolling max drawdown
    rolling_max = strat_df['Total_Value'].cummax()
    drawdown = (strat_df['Total_Value'] - rolling_max) / rolling_max * 100

    # Generate plot   
    plt.figure(figsize=(10, 5), facecolor = "#F5F5F5")
    
    # Plotting data
    plt.plot(strat_df.index, drawdown, label = "Trend Following Model Drawdown", linestyle = "-", color = "red", linewidth = 1)
    for idx, fund in enumerate(fund_list):
        # Calculate rolling max drawdown
        rolling_max = buy_hold[fund + "_Value"].cummax()
        drawdown = (buy_hold[fund + "_Value"] - rolling_max) / rolling_max * 100
        plt.plot(buy_hold.index, drawdown, label = fund + " Drawdown", linestyle = "-", color = colors[idx % len(colors)], linewidth = 1)
    
    # Set X axis
    # x_tick_spacing = 5  # Specify the interval for x-axis ticks
    # plt.gca().xaxis.set_major_locator(MultipleLocator(x_tick_spacing))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xlabel("Year", fontsize = 9)
    plt.xticks(rotation = 0, fontsize = 7)

    # Set Y axis
    min_value = drawdown.min()
    y_tick_spacing = round((-1 * min_value / 15), 0) # Specify the interval for y-axis ticks
    # y_tick_spacing = 5  # Specify the interval for y-axis ticks
    plt.gca().yaxis.set_major_locator(MultipleLocator(y_tick_spacing))
    # plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))) # Adding commas to y-axis labels
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:.0f}'.format(x))) # Adding 2 decimal places to y-axis labels
    plt.ylabel("Drawdown (%)", fontsize = 9)
    plt.yticks(fontsize = 7)

    # Set title, etc.
    plt.title("Trend Following Model Portfolio Drawdown", fontsize = 12)
    
    # Set the grid & legend
    plt.tight_layout()
    plt.grid(True)
    plt.legend(fontsize=7)

    # Save the figure
    # plt.savefig('Crypto_Trend_Following/Portfolio_Drawdown.png', dpi=300, bbox_inches='tight')
    # plt.savefig("Portfolio_Drawdown.png", dpi = 300, bbox_inches = "tight")

    # Display the plot
    return plt.show()

def plot_price(strat_df, fund):   
    # Generate plot   
    plt.figure(figsize=(10, 5), facecolor = "#F5F5F5")
    
    # Plotting data
    plt.plot(strat_df.index, strat_df[fund + "_Close"], label= fund + " Close Price", linestyle = "-", color = "blue", linewidth = 1)
    
    # Set X axis
    # x_tick_spacing = 5  # Specify the interval for x-axis ticks
    # plt.gca().xaxis.set_major_locator(MultipleLocator(x_tick_spacing))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xlabel("Year", fontsize = 9)
    plt.xticks(rotation = 0, fontsize = 7)

    # Set Y axis
    max_value = strat_df[fund + "_Close"].max()
    if max_value >= 1500000:
        round_value = -5
    elif max_value >= 150000:
        round_value = -4    
    elif max_value >= 15000:
        round_value = -3
    elif max_value >= 1500:
        round_value = -2
    else:
        round_value = -1
    y_tick_spacing = round(max_value / 15, round_value) # Specify the interval for y-axis ticks
    plt.gca().yaxis.set_major_locator(MultipleLocator(y_tick_spacing))
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))) # Adding commas to y-axis labels
    plt.ylabel("Price ($)", fontsize = 9)
    plt.yticks(fontsize = 7)

    # Set title, etc.
    plt.title(fund + " Close/Adjusted Close Price", fontsize = 12)
    
    # Set the grid & legend
    plt.tight_layout()
    plt.grid(True)
    plt.legend(fontsize=7)

    # Save the figure
    # plt.savefig('Crypto_Trend_Following/Portfolio_Values.png', dpi=300, bbox_inches='tight')
    # plt.savefig(fund + "_Price_History.png", dpi=300, bbox_inches='tight')

    # Display the plot
    return plt.show()

def plot_cum_trans_costs(strat_df):   
    # Generate plot   
    plt.figure(figsize=(10, 5), facecolor = "#F5F5F5")
    
    # Plotting data
    plt.plot(strat_df.index, strat_df['Cum_Total_Transaction_Costs'], label= "Trend Following Model Cumulative Total Transaction Costs", linestyle = "-", color = "red", linewidth = 1)
    
    # Set X axis
    # x_tick_spacing = 5  # Specify the interval for x-axis ticks
    # plt.gca().xaxis.set_major_locator(MultipleLocator(x_tick_spacing))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xlabel("Year", fontsize = 9)
    plt.xticks(rotation = 0, fontsize = 7)

    # Set Y axis
    max_value = strat_df['Cum_Total_Transaction_Costs'].max()
    if max_value >= 1500000:
        round_value = -5
    elif max_value >= 150000:
        round_value = -4    
    elif max_value >= 15000:
        round_value = -3
    elif max_value >= 1500:
        round_value = -2
    else:
        round_value = -1
    y_tick_spacing = round(max_value / 15, round_value) # Specify the interval for y-axis ticks
    plt.gca().yaxis.set_major_locator(MultipleLocator(y_tick_spacing))
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))) # Adding commas to y-axis labels
    plt.ylabel("Total Value ($)", fontsize = 9)
    plt.yticks(fontsize = 7)

    # Set title, etc.
    plt.title("Trend Following Model Cumulative Total Transaction Costs", fontsize = 12)
    
    # Set the grid & legend
    plt.tight_layout()
    plt.grid(True)
    plt.legend(fontsize=7)

    # Save the figure
    # plt.savefig('Crypto_Trend_Following/Portfolio_Values.png', dpi=300, bbox_inches='tight')
    # plt.savefig("_Price_History.png", dpi=300, bbox_inches='tight')

    # Display the plot
    return plt.show()