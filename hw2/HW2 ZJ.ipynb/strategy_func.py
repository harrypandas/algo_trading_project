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

def strategy(sma_list, num_sma, num_funds, fund_list, starting_cash, cash_contrib, rebal_interval, trans_cost_pct):
    # Read in the combined price data for the strategy from an existing excel file
    plan_name = '_'.join(fund_list)
    file = plan_name + "_Combined_Data.xlsx"
    # location = "Crypto_Trend_Following/Data/" + file
    location = "Data/" + file
    df = pd.read_excel(location, sheet_name = 'data').set_index('Date')
    df.reset_index(inplace=True)

    # Calculate moving averages for each asset and for each interval and the buy/sell signals
    for fund in fund_list:
        for interval in sma_list:
            # Calculate moving average
            # df[fund + "_" + interval] = df[fund + "_Close"].rolling(int(interval)).mean() # For simple moving averages
            df[fund + "_" + interval] = df[fund + "_Close"].ewm(span=int(interval), adjust=False).mean() # For exponential moving averages

            # Calculate difference between month end price and moving average
            df[fund + "-" + fund + "_" + interval] = df[fund + "_Close"] - df[fund + "_" + interval]

            # Generate binary output on whether month end price is greater or less than moving average
            df[fund + "_" + interval + "_Sig"] = np.where(df[fund + "_Close"] > df[fund + "_" + interval], 1, 0)

            # Generate diff from binary output above
            df[fund + "_" + interval + "_Sig_Change"] = df[fund + "_" + interval + "_Sig"].diff()

            # Generate order from diff above
            col         = fund + "_" + interval + "_Sig_Change"
            conditions  = [ df[col] == 1, df[col] == 0, df[col] == -1 ]
            choices     = [ 'BUY', 'HOLD', 'SELL' ]

            # df[fund + "_" + interval + "_Action"] = np.select(conditions, choices, default=np.nan)
            df[fund + "_" + interval + "_Action"] = np.select(conditions, choices, default=None)

    '''
    Column order:
    df[fund + "_BA_Shares"]
    df[fund + "_BA_$_Invested"]
    df['Total_BA_$_Invested']
    df['Cash_BA']
    df['Contribution']
    df[fund + "_AA_%_Invested"]
    df[fund + "_AA_$_Invested"]
    df[fund + "_AA_Shares"]
    df[fund + "_Notional"]
    df['Total_AA_%_Invested']
    df['Total_AA_$_Invested']
    df['Total_Transaction_Costs']
    df['Cash_%']
    df['Cash_AA']
    df['Total_Value']
    '''

    # Create columns for each fund for before action (BA) shares and $ invested
    for fund in fund_list:
        df[fund + "_BA_Shares"] = 0
        df[fund + "_BA_$_Invested"] = 0

    # Create columns for before action (BA) total $ invested, before action (BA) cash balance, and contribution
    df['Total_BA_$_Invested'] = 0
    df['Cash_BA'] = starting_cash
    df['Contribution'] = 0
    # df['Contribution'] = cash_contrib

    # Calculate the columns for each fund for after action (AA) % invested, $ invested, and shares
    for fund in fund_list:
        total_sum = 0
        
        for interval in sma_list:
            total_sum = total_sum + df[fund + "_" + interval + "_Sig"]

        for interval in sma_list:
            df.drop(columns = {fund + "_" + interval + "_Sig", fund + "_" + interval + "_Sig_Change"}, inplace = True)

        df[fund + "_AA_%_Invested"] = total_sum / num_funds / num_sma
        df[fund + "_AA_$_Invested"] = 0
        df[fund + "_AA_Shares"] = 0
        df[fund + "_Notional"] = 0

    # Create columns for after action (AA) total % invested, total $ invested, cash, cash % of portfolio, and total portfolio value
    df['Total_AA_%_Invested'] = 0
    df['Total_AA_$_Invested'] = 0
    df['Total_Transaction_Costs'] = 0
    df['Cash_%'] = 0
    df['Cash_AA'] = 0
    df['Total_Value'] = 0

    # Create variables
    Total_BA_Invested = 0
    Cash_BA = 0
    Total_AA_Percent_Invested = 0
    Total_AA_Invested = 0
    Total_Transaction_Costs = 0

    # Iterate through the dataframe to run strategy
    for index, row in df.iterrows():

        # Ensure there's a previous row to reference
        if index > 0 and index % rebal_interval != 0:
            for fund in fund_list:
                # Fill in BA from previous row AA for each fund
                df.at[index, fund + "_BA_Shares"] = df.at[index - 1, fund + "_AA_Shares"]

                # Calc BA $ invested for each fund
                df.at[index, fund + "_BA_$_Invested"] = row[fund + "_Close"] * df.at[index, fund + "_BA_Shares"]

                # Calc total BA $ invested
                Total_BA_Invested = Total_BA_Invested + df.at[index, fund + "_BA_$_Invested"]

            df.at[index, 'Total_BA_$_Invested'] = Total_BA_Invested
            Total_BA_Invested = 0 # Reset variable value to 0

            for fund in fund_list:
                # Cash_BA = Cash_BA + df.at[index, fund + "_BA_Shares"] * df.at[index, fund + "_Dividend"] # Includes dividends if they are paid
                Cash_BA = Cash_BA # Does not include dividends

            # Calc BA cash balance
            df.at[index, 'Cash_BA'] = Cash_BA + df.at[index - 1, 'Cash_AA']
            Cash_BA = 0 # Reset variable value to 0

            # Calc AA $ invested, shares, % invested, and notional for each fund
            # The AA values are the same as BA values because the portfolio is only rebalanced every rebal_interval days
            for fund in fund_list:
                df.at[index, fund + "_AA_$_Invested"] = df.at[index, fund + "_BA_$_Invested"]
                df.at[index, fund + "_AA_Shares"] = df.at[index, fund + "_BA_Shares"]
                df.at[index, fund + "_Notional"] = df.at[index, fund + "_AA_$_Invested"] - df.at[index, fund + "_BA_$_Invested"]

                # Calc total AA % invested
                Total_AA_Percent_Invested = Total_AA_Percent_Invested + df.at[index, fund + "_AA_%_Invested"]

                # Calc total AA $ invested
                Total_AA_Invested = Total_AA_Invested + df.at[index, fund + "_AA_$_Invested"]

                # Calc total transaction costs
                Total_Transaction_Costs = Total_Transaction_Costs + (trans_cost_pct * abs(df.at[index, fund + "_Notional"]))

            df.at[index, 'Total_AA_%_Invested'] = Total_AA_Percent_Invested
            Total_AA_Percent_Invested = 0 # Reset variable to 0
            
            df.at[index, 'Total_AA_$_Invested'] = Total_AA_Invested
            Total_AA_Invested = 0 # Reset variable to 0

            df.at[index, 'Total_Transaction_Costs'] = Total_Transaction_Costs
            Total_Transaction_Costs = 0 # Reset variable to 0

            # Calc cash AA
            df.at[index, 'Cash_AA'] = (df.at[index, 'Total_BA_$_Invested'] + 
                                       df.at[index, 'Cash_BA'] + 
                                       df.at[index, 'Contribution'] - 
                                       df.at[index, 'Total_AA_$_Invested'] - 
                                       df.at[index, 'Total_Transaction_Costs'])
            
            # Calc total value
            df.at[index, 'Total_Value'] = df.at[index, 'Total_AA_$_Invested'] + df.at[index, 'Cash_AA']

            # Calc AA cash %
            df.at[index, 'Cash_%'] = df.at[index, 'Cash_AA'] / df.at[index, 'Total_Value']

        elif index > 0 and index % rebal_interval == 0:
            for fund in fund_list:
                # Fill in BA from previous row AA for each fund
                df.at[index, fund + "_BA_Shares"] = df.at[index - 1, fund + "_AA_Shares"]

                # Calc BA $ invested for each fund
                df.at[index, fund + "_BA_$_Invested"] = row[fund + "_Close"] * df.at[index, fund + "_BA_Shares"]

                # Calc total BA $ invested
                Total_BA_Invested = Total_BA_Invested + df.at[index, fund + "_BA_$_Invested"]

            df.at[index, 'Total_BA_$_Invested'] = Total_BA_Invested
            Total_BA_Invested = 0 # Reset variable value to 0

            for fund in fund_list:
                # Cash_BA = Cash_BA + df.at[index, fund + "_BA_Shares"] * df.at[index, fund + "_Dividend"] # Includes dividends if they are paid
                Cash_BA = Cash_BA # Does not include dividends

            # Calc BA cash balance
            df.at[index, 'Cash_BA'] = Cash_BA + df.at[index - 1, 'Cash_AA']
            Cash_BA = 0 # Reset variable value to 0

            # Calc AA $ invested, shares, % invested, and notional for each fund
            # The AA values are different from the BA values because the portfolio is only rebalanced every rebal_interval days
            for fund in fund_list:
                df.at[index, fund + "_AA_$_Invested"] = row[fund + "_AA_%_Invested"] * (df.at[index, 'Total_BA_$_Invested'] + df.at[index, 'Cash_BA'] + df.at[index, 'Contribution'])
                df.at[index, fund + "_AA_Shares"] = df.at[index, fund + "_AA_$_Invested"] / row[fund + "_Close"]
                df.at[index, fund + "_Notional"] = df.at[index, fund + "_AA_$_Invested"] - df.at[index, fund + "_BA_$_Invested"]

                # Calc total AA % invested
                Total_AA_Percent_Invested = Total_AA_Percent_Invested + df.at[index, fund + "_AA_%_Invested"]

                # Calc total AA $ invested
                Total_AA_Invested = Total_AA_Invested + df.at[index, fund + "_AA_$_Invested"]

                # Calc total transaction costs
                Total_Transaction_Costs = Total_Transaction_Costs + (trans_cost_pct * abs(df.at[index, fund + "_Notional"]))

            df.at[index, 'Total_AA_%_Invested'] = Total_AA_Percent_Invested
            Total_AA_Percent_Invested = 0 # Reset variable to 0
            
            df.at[index, 'Total_AA_$_Invested'] = Total_AA_Invested
            Total_AA_Invested = 0 # Reset variable to 0

            df.at[index, 'Total_Transaction_Costs'] = Total_Transaction_Costs
            Total_Transaction_Costs = 0 # Reset variable to 0

            # Calc cash AA
            df.at[index, 'Cash_AA'] = (df.at[index, 'Total_BA_$_Invested'] + 
                                       df.at[index, 'Cash_BA'] + 
                                       df.at[index, 'Contribution'] - 
                                       df.at[index, 'Total_AA_$_Invested'] - 
                                       df.at[index, 'Total_Transaction_Costs'])
            
            # Calc total value
            df.at[index, 'Total_Value'] = df.at[index, 'Total_AA_$_Invested'] + df.at[index, 'Cash_AA']

            # Calc AA cash %
            df.at[index, 'Cash_%'] = df.at[index, 'Cash_AA'] / df.at[index, 'Total_Value']

        # If this is the first row
        elif index == 0:
            df.at[index, 'Cash_AA'] = df.at[index, 'Cash_BA']
            df.at[index, 'Total_Value'] = df.at[index, 'Total_AA_$_Invested'] + df.at[index, 'Cash_AA']
            df.at[index, 'Cash_%'] = df.at[index, 'Cash_AA'] / df.at[index, 'Total_Value']

        else:
            print(index)

    # Calculate daily return
    df['Return'] = df['Total_Value'].pct_change()

    # Calculate cumulative return
    df['Cumulative_Return'] = (1 + df['Return']).cumprod()

    # Calculate cumulative total transaction costs
    df['Cum_Total_Transaction_Costs'] = df['Total_Transaction_Costs'].cumsum()
    
    # Export strategy to excel
    file = plan_name + "_Strategy.xlsx"
    # location = "Crypto_Trend_Following/" + file
    location = file
    df.to_excel(location, sheet_name='data')
    print(f"Strategy complete for {plan_name}.")
    return df