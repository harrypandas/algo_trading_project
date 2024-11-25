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
from pathlib import Path

warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x: '%.4f' % x)

def combine_data(fund_list, timeframe):
    fundlist = fund_list.copy()
    plan_name = '_'.join(fundlist)

    ###
    file = plan_name + "_Combined_Data.xlsx"
    location = "Data/" + file

    # Check if the file already exists
    file_path = Path(location)
    if file_path.exists():
        print(f"The file '{location}' already exists. Combine data aborted.")
        return
    else:  
    ###
        max_fund_length = 0
        max_fund_length_name = None
        
        for fund in fundlist:
            file = fund + ".xlsx"
            # location = "Crypto_Trend_Following/Data/" + file
            location = "Data/" + file
            fund_data = pd.read_excel(location, sheet_name = "data") # dataframe for the fund
            fund_data_len = fund_data.shape[0]
            if fund_data_len > max_fund_length:
                max_fund_length = fund_data_len
                max_fund_length_name = fund
            else:
                pass
        
        file = max_fund_length_name + ".xlsx"
        # location = "Crypto_Trend_Following/Data/" + file
        location = "Data/" + file
        
        all_data = pd.read_excel(location, sheet_name = 'data').set_index('Date')
        all_data = all_data[['open', 'high', 'low', 'close']]
        all_data.rename(columns = {'open': max_fund_length_name + '_Open',
                                'high': max_fund_length_name + '_High', 
                                'low': max_fund_length_name + '_Low', 
                                'close': max_fund_length_name + '_Close'}, inplace = True)
        
        fundlist.remove(max_fund_length_name)
        
        for fund in fundlist:
            file = fund + ".xlsx"
            # location = "Crypto_Trend_Following/Data/" + file
            location = "Data/" + file
            fund_data = pd.read_excel(location, sheet_name = 'data').set_index('Date')
            fund_data = fund_data[['open', 'high', 'low', 'close']]
            fund_data.rename(columns = {'open': fund + '_Open',
                                        'high': fund + '_High', 
                                        'low': fund + '_Low', 
                                        'close': fund + '_Close'}, inplace = True)
            all_data = all_data.join(fund_data)
        
        all_data.dropna(inplace = True)

        if timeframe == 'Monthly':
            all_data = all_data.resample('ME').last()
        elif timeframe == 'Weekly':
            all_data = all_data.resample('W').last()
        
        fundlist.append(max_fund_length_name)
            
        # Define the file name and location
        file = plan_name + "_Combined_Data.xlsx"
        location = "Data/" + file

        ###
        # Ensure the target directory exists
        os.makedirs("Data", exist_ok=True)

        # Check if the file already exists
        file_path = Path(location)
        if file_path.exists():
            print(f"The file '{location}' already exists. Data export aborted.")
        else:
            try:
                # Export to Excel
                all_data.to_excel(location, sheet_name='data')
                print(f"Data exported successfully to {location}")
            except Exception as e:
                print(f"An error occurred while exporting: {e}")
        ###

        # print(all_data)
        print(f"Combine data complete for {plan_name}.")
        return all_data

