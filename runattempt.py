import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
import regex as re
import matplotlib.pyplot as plt
from datetime import timedelta
import statsmodels.api as sm
import gc  # For garbage collection
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import os
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import warnings
import multiprocessing
#multiprocessing.set_start_method('spawn')
warnings.filterwarnings('ignore')

def calculate_metrics(df, returns_column, benchmark_returns_column=None):
    """
    Calculate performance metrics for a given returns series.

    Parameters:
    - df: DataFrame containing the returns data.
    - returns_column: Column name for the strategy returns.
    - benchmark_returns_column: Column name for the benchmark returns (if any).

    Returns:
    - metrics: Dictionary of calculated metrics.
    """
    # Calculate cumulative returns
    df['Cumulative_Returns'] = (1 + df[returns_column]).cumprod()

    # Calculate total time period in years
    total_days = (df.index[-1] - df.index[0]).total_seconds() / (3600 * 24)
    total_years = total_days / 365.25  # Accounting for leap years

    ## Metrics Calculation ##

    # Total Return
    total_return = df['Cumulative_Returns'].iloc[-1] - 1
    annualized_return = (1 + total_return) ** (1 / total_years) - 1

    # Number of Transactions (if 'Trade' column exists)
    if 'Trade' in df.columns and returns_column == 'Strategy_Returns':
        num_transactions = df['Trade'].sum()
    else:
        num_transactions = np.nan  # Not applicable for benchmark

    # Average Gain/Loss per Transaction
    if 'Trade' in df.columns and returns_column == 'Strategy_Returns':
        trades = df[df['Trade'] == 1]
        trade_indices = trades.index
        trade_profits = []

        for i in range(1, len(trade_indices)):
            start_index = trade_indices[i - 1]
            end_index = trade_indices[i]
            profit = df.loc[end_index, 'Position'] - df.loc[start_index, 'Position']
            trade_profits.append(profit)
        if trade_profits:
            avg_gain_loss = np.mean(trade_profits)
        else:
            avg_gain_loss = 0
    else:
        avg_gain_loss = np.nan  # Not applicable for benchmark

    # Maximum Drawdown
    rolling_max = df['Cumulative_Returns'].cummax()
    drawdown = (df['Cumulative_Returns'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Annualized Sharpe Ratio
    risk_free_rate = 0  # Assuming zero risk-free rate
    periods_per_year = 365.25 * 24 * 60  # Total minutes in a year
    mean_return = df[returns_column].mean() * periods_per_year
    std_return = df[returns_column].std() * np.sqrt(periods_per_year)
    if std_return != 0:
        sharpe_ratio = (mean_return - risk_free_rate) / std_return
    else:
        sharpe_ratio = np.nan

    # Tracking Error and Information Ratio (if benchmark returns are provided)
    if benchmark_returns_column:
        df['Excess_Returns'] = df[returns_column] - df[benchmark_returns_column]
        tracking_error = df['Excess_Returns'].std() * np.sqrt(periods_per_year)
        mean_excess_return = df['Excess_Returns'].mean() * periods_per_year
        if tracking_error != 0:
            information_ratio = mean_excess_return / tracking_error
        else:
            information_ratio = np.nan
    else:
        tracking_error = np.nan
        information_ratio = np.nan

    # Treynor Ratio and Jensen's Alpha (Require Beta estimation)
    if benchmark_returns_column:
        df.dropna(subset=[returns_column, benchmark_returns_column], inplace=True)
        X = df[benchmark_returns_column]
        y = df[returns_column]
        X = sm.add_constant(X)  # Adds a constant term to the predictor
        model = sm.OLS(y, X).fit()
        beta = model.params[benchmark_returns_column]

        if beta != 0:
            treynor_ratio = (mean_return - risk_free_rate) / beta
        else:
            treynor_ratio = np.nan
        mean_benchmark_return = df[benchmark_returns_column].mean() * periods_per_year
        jensens_alpha = mean_return - (risk_free_rate + beta * (mean_benchmark_return - risk_free_rate))
    else:
        beta = np.nan
        treynor_ratio = np.nan
        jensens_alpha = np.nan

    ## Metrics Summary ##
    metrics = {
        'Total Return': f"{total_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}",
        'Number of Transactions': f"{int(num_transactions) if not np.isnan(num_transactions) else 'N/A'}",
        'Average Gain/Loss per Transaction': f"${avg_gain_loss:,.2f}" if not np.isnan(avg_gain_loss) else 'N/A',
        'Maximum Drawdown': f"{max_drawdown:.2%}",
        'Annualized Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Tracking Error': f"{tracking_error:.2%}" if not np.isnan(tracking_error) else 'N/A',
        'Information Ratio': f"{information_ratio:.2f}" if not np.isnan(information_ratio) else 'N/A',
        'Treynor Ratio': f"{treynor_ratio:.2f}" if not np.isnan(treynor_ratio) else 'N/A',
        'Jensen\'s Alpha': f"{jensens_alpha:.2%}" if not np.isnan(jensens_alpha) else 'N/A'
    }

    return metrics


def analyze_strategies(df):
    # Ensure 'Position' and 'Pos_Benchmark' are numeric
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    df['Pos_Benchmark'] = pd.to_numeric(df['Pos_Benchmark'], errors='coerce')

    # Calculate Strategy Returns
    df['Strategy_Returns'] = df['Position'].pct_change()

    # Calculate Benchmark Returns
    df['Benchmark_Returns'] = df['Pos_Benchmark'].pct_change()

    # Metrics for Strategy
    strategy_metrics = calculate_metrics(df.copy(), 'Strategy_Returns', 'Benchmark_Returns')

    # Metrics for Benchmark
    benchmark_metrics = calculate_metrics(df.copy(), 'Benchmark_Returns')

    # Combine metrics into a DataFrame for better formatting
    metrics_df = pd.DataFrame({
        'Metric': strategy_metrics.keys(),
        'Strategy': strategy_metrics.values(),
        'Benchmark': benchmark_metrics.values()
    })

    # Set 'Metric' as the index
    metrics_df.set_index('Metric', inplace=True)

    ## Visualizations ##
    # Cumulative Performance Chart
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, (1 + df['Strategy_Returns']).cumprod(), label='Strategy')
    plt.plot(df.index, (1 + df['Benchmark_Returns']).cumprod(), label='Benchmark')
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

    # Returns Distribution for Strategy
    plt.figure(figsize=(10, 5))
    df['Strategy_Returns'].hist(bins=50)
    plt.title('Strategy Returns Distribution')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.show()

    return metrics_df

def simulate_trading_strategy(
    df,
    target_variable,
    initial_capital=10_000_000,
    transaction_cost=0.001,  # 0.1% transaction cost for market orders
    start_date=None,
    end_date=None,
    strategy_params=None
):
    """
    Simulate a trading strategy with added stop loss and freeze window functionality.
    Once the portfolio experiences a drawdown greater than the stop_loss_pct from its max value,
    we immediately sell all BTC and then freeze trading for 'freeze_window' minutes.

    Parameters:
    - df: DataFrame with prices and predictions.
    - target_variable: Name of the target variable (e.g., 'BTC_Future_Direction_5m').
    - initial_capital: Starting capital in USD.
    - transaction_cost: Transaction cost proportion.
    - start_date: Filter start date (string 'YYYY-MM-DD').
    - end_date: Filter end date (string 'YYYY-MM-DD').
    - strategy_params: Dict of strategy parameters.
    - stop_loss_pct: Drawdown percentage to trigger stop loss.
    - freeze_window: Minutes to freeze trading after stop loss is triggered.

    Returns:
    - df: DataFrame with strategy results.
    - performance_metrics: Dictionary of performance metrics.
    - trade_history_df: DataFrame of trade history.
    """

    # Set default strategy parameters if not provided
    if strategy_params is None:
        strategy_params = {
            'investment_fraction': 1.0,     # Fraction of capital to invest (if no fixed amount)
            'buy_investment_amount': None,  # Fixed USD amount to invest when buying
            'sell_investment_amount': None, # Fixed USD amount to sell from holdings
            'use_limit_orders': False,      # Whether to use limit orders
            'buy_threshold': 0.6,           # Probability threshold to buy
            'sell_threshold': 0.4,          # Probability threshold to sell
            'limit_order_buffer': 0.001,    # Price buffer for limit orders
            'limit_order_duration': 5,       # Duration in minutes for limit order validity
            'stop_loss_pct' : 0.10,      # e.g., 0.10 for 10%
            'freeze_window' : 60         # Number of minutes to freeze trading after stop loss event
        }

    # Filter data by date range
    if start_date is not None and end_date is not None:
        df = df[(df.index >= start_date) & (df.index <= end_date)].copy()
    else:
        df = df.copy()

    # Initialize variables
    stop_loss_pct = strategy_params['stop_loss_pct']
    freeze_window = strategy_params['freeze_window']
    capital = initial_capital
    btc_balance = 0.0
    usd_balance = capital
    position_history = []
    portfolio_value_history = []
    trade_list = []
    benchmark_position_history = []
    signal_generated_list = []
    strategy_returns = []
    holdings_history = []
    transaction_costs_total = 0.0

    # Pending orders
    pending_orders = []
    pending_market_orders = []

    # Signal and trade history tracking
    signal_active = False
    signal_start_idx = None
    signal_info = None
    trade_history_records = []

    # Stop loss and freeze logic variables
    max_portfolio_value = initial_capital
    freeze_until_time = df.index[0] - pd.Timedelta(minutes=1)  # Initialize in the past, so not frozen initially

    for idx in range(len(df)):
        row = df.iloc[idx]
        date = df.index[idx]

        btc_high = row['BTC_High']
        btc_low = row['BTC_Low']

        # Initialize iteration variables
        trade_occurred = 0
        signal_generated = 0
        trade_action = 'hold'

        # Process pending market orders from previous iteration
        if idx > 0:
            orders_to_remove = []
            for order in pending_market_orders:
                asset = order['asset']
                order_type = order['type']

                # Execute market orders at current iteration's prices
                if asset == 'BTC':
                    high_price = btc_high
                    low_price = btc_low
                else:
                    continue

                if order_type == 'buy':
                    execution_price = high_price
                    # Determine investment amount
                    if strategy_params['buy_investment_amount'] is not None:
                        investment_amount = min(usd_balance, strategy_params['buy_investment_amount'])
                    else:
                        investment_fraction = order['investment_fraction']
                        investment_amount = usd_balance * investment_fraction

                    if investment_amount <= 0:
                        continue
                    transaction_fee = investment_amount * transaction_cost
                    quantity = (investment_amount - transaction_fee) / execution_price
                    usd_balance -= investment_amount
                    btc_balance += quantity
                    trade_action = f'buy_{asset.lower()}'

                elif order_type == 'sell':
                    execution_price = low_price
                    # Determine sell amount
                    btc_value = btc_balance * execution_price
                    if strategy_params['sell_investment_amount'] is not None:
                        amount_to_sell_usd = min(btc_value, strategy_params['sell_investment_amount'])
                        quantity = amount_to_sell_usd / execution_price
                    else:
                        investment_fraction = order['investment_fraction']
                        quantity = btc_balance * investment_fraction

                    if quantity <= 0 or btc_balance <= 0:
                        continue
                    quantity = min(quantity, btc_balance)
                    proceeds = quantity * execution_price
                    transaction_fee = proceeds * transaction_cost
                    btc_balance -= quantity
                    usd_balance += proceeds - transaction_fee
                    trade_action = f'sell_{asset.lower()}'

                transaction_costs_total += transaction_fee
                trade_occurred = 1
                orders_to_remove.append(order)

                if signal_active:
                    # Record signal to trade history
                    signal_end_idx = idx
                    signal_period = df.iloc[signal_start_idx:signal_end_idx + 1]
                    for i in range(len(signal_period)):
                        signal_row = signal_period.iloc[i]
                        record = {
                            'Date': signal_row.name,
                            'Signal Generated': 1 if i == 0 else 0,
                            'Action': trade_action if i == len(signal_period) - 1 else 'waiting',
                            'Asset': asset,
                            'Order Type': order_type,
                            'Execution Price': execution_price if i == len(signal_period) - 1 else None,
                            'Quantity': quantity if i == len(signal_period) - 1 else None,
                            'Transaction Fee': transaction_fee if i == len(signal_period) - 1 else None,
                            'Holdings': {'BTC': btc_balance, 'USD': usd_balance},
                            'Portfolio Value': btc_balance * btc_high + usd_balance
                        }
                        trade_history_records.append(record)
                    signal_active = False
                    signal_start_idx = None
                    signal_info = None

            for order in orders_to_remove:
                pending_market_orders.remove(order)

        # Get prediction probability
        prediction = row[target_variable]
        proba_column = f'{target_variable}_Proba'
        if proba_column in df.columns:
            prediction_proba = row[proba_column]
        else:
            prediction_proba = 1.0 if prediction == 1 else 0.0

        # Process pending limit orders
        orders_to_remove = []
        for order in pending_orders:
            order_time = order['time']
            time_diff = (date - order_time).total_seconds() / 60
            if time_diff >= strategy_params['limit_order_duration']:
                # Cancel expired order
                orders_to_remove.append(order)
                if signal_active:
                    signal_active = False
                    signal_start_idx = None
                    signal_info = None
        for order in orders_to_remove:
            pending_orders.remove(order)

        # Process limit orders for execution at T+1
        if idx < len(df) - 1:
            next_row = df.iloc[idx + 1]
            next_high_price = next_row['BTC_High']
            next_low_price = next_row['BTC_Low']

            orders_to_remove = []
            for order in pending_orders:
                asset = order['asset']
                order_type = order['type']
                limit_price = order['limit_price']

                if asset == 'BTC':
                    price_reached = False
                    if order_type == 'buy' and next_low_price <= limit_price:
                        price_reached = True
                    elif order_type == 'sell' and next_high_price >= limit_price:
                        price_reached = True

                    if price_reached:
                        # Execute limit order at limit price
                        execution_price = limit_price

                        if order_type == 'buy':
                            if strategy_params['buy_investment_amount'] is not None:
                                investment_amount = min(usd_balance, strategy_params['buy_investment_amount'])
                            else:
                                investment_fraction = strategy_params['investment_fraction']
                                investment_amount = usd_balance * investment_fraction

                            if investment_amount <= 0:
                                continue
                            transaction_fee = 0.0
                            quantity = investment_amount / execution_price
                            usd_balance -= investment_amount
                            btc_balance += quantity
                            trade_action = f'limit_buy_{asset.lower()}'

                        elif order_type == 'sell':
                            btc_value = btc_balance * execution_price
                            if strategy_params['sell_investment_amount'] is not None:
                                amount_to_sell_usd = min(btc_value, strategy_params['sell_investment_amount'])
                                quantity = amount_to_sell_usd / execution_price
                            else:
                                investment_fraction = strategy_params['investment_fraction']
                                quantity = btc_balance * investment_fraction

                            if quantity <= 0 or btc_balance <= 0:
                                continue
                            quantity = min(quantity, btc_balance)
                            proceeds = quantity * execution_price
                            transaction_fee = 0.0
                            btc_balance -= quantity
                            usd_balance += proceeds
                            trade_action = f'limit_sell_{asset.lower()}'

                        transaction_costs_total += transaction_fee
                        trade_occurred = 1
                        orders_to_remove.append(order)

                        if signal_active:
                            signal_end_idx = idx + 1
                            signal_period = df.iloc[signal_start_idx:signal_end_idx + 1]
                            for i in range(len(signal_period)):
                                signal_row = signal_period.iloc[i]
                                record = {
                                    'Date': signal_row.name,
                                    'Signal Generated': 1 if i == 0 else 0,
                                    'Action': trade_action if i == len(signal_period) - 1 else 'waiting',
                                    'Asset': asset,
                                    'Order Type': order_type,
                                    'Execution Price': execution_price if i == len(signal_period) - 1 else None,
                                    'Quantity': quantity if i == len(signal_period) - 1 else None,
                                    'Transaction Fee': transaction_fee if i == len(signal_period) - 1 else None,
                                    'Holdings': {'BTC': btc_balance, 'USD': usd_balance},
                                    'Portfolio Value': btc_balance * next_high_price + usd_balance
                                }
                                trade_history_records.append(record)
                            signal_active = False
                            signal_start_idx = None
                            signal_info = None

            for order in orders_to_remove:
                pending_orders.remove(order)

        # Update holdings and portfolio value before making decisions
        holdings = {'BTC': btc_balance, 'USD': usd_balance}
        total_value = btc_balance * btc_high + usd_balance

        # Check if we are in freeze period
        in_freeze = date < freeze_until_time

        # Only proceed with signals if not in freeze
        if not in_freeze:
            # Decision-making logic based on target variable
            signal_triggered = False
            if strategy_params['use_limit_orders']:
                # Limit order logic
                if target_variable == 'BTC_Future_Direction_5m':
                    if prediction_proba >= strategy_params['buy_threshold']:
                        signal_generated = 1
                        signal_triggered = True
                        if usd_balance > 0:
                            if not any(o['asset'] == 'BTC' and o['type'] == 'buy' for o in pending_orders):
                                limit_price = btc_low * (1 - strategy_params['limit_order_buffer'])
                                pending_orders.append({
                                    'asset': 'BTC',
                                    'type': 'buy',
                                    'limit_price': limit_price,
                                    'time': date
                                })
                                if not signal_active:
                                    signal_active = True
                                    signal_start_idx = idx
                                    signal_info = {'asset': 'BTC', 'type': 'buy'}

                    elif prediction_proba <= strategy_params['sell_threshold']:
                        signal_generated = 1
                        signal_triggered = True
                        if btc_balance > 0:
                            if not any(o['asset'] == 'BTC' and o['type'] == 'sell' for o in pending_orders):
                                limit_price = btc_high * (1 + strategy_params['limit_order_buffer'])
                                pending_orders.append({
                                    'asset': 'BTC',
                                    'type': 'sell',
                                    'limit_price': limit_price,
                                    'time': date
                                })
                                if not signal_active:
                                    signal_active = True
                                    signal_start_idx = idx
                                    signal_info = {'asset': 'BTC', 'type': 'sell'}

            else:
                # Market order logic
                if target_variable == 'BTC_Future_Direction_5m':
                    if prediction_proba >= strategy_params['buy_threshold']:
                        signal_generated = 1
                        signal_triggered = True
                        if usd_balance > 0:
                            pending_order = {'asset': 'BTC', 'type': 'buy'}
                            if strategy_params['buy_investment_amount'] is None:
                                pending_order['investment_fraction'] = strategy_params['investment_fraction']
                            pending_market_orders.append(pending_order)
                            if not signal_active:
                                signal_active = True
                                signal_start_idx = idx
                                signal_info = {'asset': 'BTC', 'type': 'buy'}

                    elif prediction_proba <= strategy_params['sell_threshold']:
                        signal_generated = 1
                        signal_triggered = True
                        if btc_balance > 0:
                            pending_order = {'asset': 'BTC', 'type': 'sell'}
                            if strategy_params['sell_investment_amount'] is None:
                                pending_order['investment_fraction'] = strategy_params['investment_fraction']
                            pending_market_orders.append(pending_order)
                            if not signal_active:
                                signal_active = True
                                signal_start_idx = idx
                                signal_info = {'asset': 'BTC', 'type': 'sell'}

            if signal_active and not signal_triggered and trade_occurred == 0:
                # Signal expired without trade execution
                signal_active = False
                signal_start_idx = None
                signal_info = None

        # Recompute holdings and portfolio after potential trades
        holdings = {'BTC': btc_balance, 'USD': usd_balance}
        total_value = btc_balance * btc_high + usd_balance

        # Check for stop loss trigger
        # Update max_portfolio_value if current is higher
        if total_value > max_portfolio_value:
            max_portfolio_value = total_value

        drawdown = (total_value - max_portfolio_value) / max_portfolio_value
        if drawdown < -stop_loss_pct and not in_freeze:
            # Stop loss triggered: sell all BTC immediately at current low price
            if btc_balance > 0:
                execution_price = btc_low
                proceeds = btc_balance * execution_price
                transaction_fee = proceeds * transaction_cost
                usd_balance += (proceeds - transaction_fee)
                transaction_costs_total += transaction_fee
                trade_occurred = 1
                trade_action = 'stop_loss_sell_btc'
                # Record the stop loss trade
                record = {
                    'Date': date,
                    'Signal Generated': 0,
                    'Action': trade_action,
                    'Asset': 'BTC',
                    'Order Type': 'sell',
                    'Execution Price': execution_price,
                    'Quantity': btc_balance,
                    'Transaction Fee': transaction_fee,
                    'Holdings': {'BTC': 0, 'USD': usd_balance},
                    'Portfolio Value': usd_balance
                }
                trade_history_records.append(record)

                # Clear BTC balance
                btc_balance = 0.0

            # Set freeze_until_time
            freeze_until_time = date + pd.Timedelta(minutes=freeze_window)

            # Update total_value after stop loss
            total_value = usd_balance

            # NEW LINE ADDED TO MAKE STOP LOSS RELATIVE
            # Reset the max_portfolio_value to the current portfolio value after stopping out
            max_portfolio_value = total_value

        holdings_history.append({'BTC': btc_balance, 'USD': usd_balance})
        portfolio_value_history.append(total_value)

        # Current position
        current_position = 'BTC' if btc_balance > 0 else 'USD'
        position_history.append(current_position)

        # Append trade_occurred to trade list
        if len(trade_list) < idx + 1:
            trade_list.append(trade_occurred)
        signal_generated_list.append(signal_generated)

        # Strategy returns
        if idx == 0:
            strategy_returns.append(0)
        else:
            prev_total_value = portfolio_value_history[idx - 1]
            return_pct = (total_value - prev_total_value) / prev_total_value
            strategy_returns.append(return_pct)

        # Benchmark: buy-and-hold in BTC from the start
        if idx == 0:
            benchmark_btc_balance = initial_capital / btc_high
        benchmark_position = benchmark_btc_balance * btc_high
        benchmark_position_history.append(benchmark_position)

    # Add result columns to df
    df['Trade Action'] = position_history
    df['Portfolio Value'] = portfolio_value_history
    df['Position'] = portfolio_value_history
    df['Trade'] = trade_list
    df['Pos_Benchmark'] = benchmark_position_history
    df['signal_generated'] = signal_generated_list

    # Performance metrics
    performance_metrics = {
        'Initial Capital': initial_capital,
        'Final Portfolio Value': portfolio_value_history[-1],
        'Total Return (%)': ((portfolio_value_history[-1] / initial_capital) - 1) * 100,
        'Total Transaction Costs': transaction_costs_total,
        'Number of Trades': sum(trade_list)
    }

    # Trade history DataFrame
    trade_history_df = pd.DataFrame(trade_history_records)
    if not trade_history_df.empty:
        trade_history_df['timestamp'] = pd.to_datetime(trade_history_df['Date'])
        trade_history_df.set_index('timestamp', inplace=True)
        trade_history_df.drop('Date', axis=1, inplace=True)

    return df, performance_metrics, trade_history_df


def run_simulation(params, trading_df):
    (buy_amt, sell_amt, inv_frac, limit, buy_thr, sell_thr, lob, lod, slp, fw) = params

    # Ensure that buy_threshold > sell_threshold
    if buy_thr <= sell_thr:
        return None

    # Define strategy parameters
    strategy_params = {
        'buy_investment_amount': buy_amt,
        'sell_investment_amount': sell_amt,
        'investment_fraction': inv_frac,
        'use_limit_orders': limit,
        'buy_threshold': buy_thr,
        'sell_threshold': sell_thr,
        'limit_order_buffer': lob,
        'limit_order_duration': lod,
        'stop_loss_pct': slp,
        'freeze_window': fw
    }

    # Simulate the trading strategy
    df_with_results, performance_metrics, _ = simulate_trading_strategy(
        df=trading_df[(trading_df.index >= '2022-01-01') & (trading_df.index < '2024-01-01')],
        target_variable='BTC_Future_Direction_5m',  # Change as needed
        initial_capital=10_000_000,
        transaction_cost=0.001,
        strategy_params=strategy_params
    )

    # Analyze the strategy
    df_with_results['Position'] = pd.to_numeric(df_with_results['Position'], errors='coerce')
    df_with_results['Pos_Benchmark'] = pd.to_numeric(df_with_results['Pos_Benchmark'], errors='coerce')
    df_with_results['Strategy_Returns'] = df_with_results['Position'].pct_change()
    df_with_results['Benchmark_Returns'] = df_with_results['Pos_Benchmark'].pct_change()

    # Calculate performance metrics
    strategy_metrics = calculate_metrics(df_with_results.copy(), 'Strategy_Returns', 'Benchmark_Returns')

    # Store the performance metrics and parameters
    result = {
        'buy_investment_amount': buy_amt,
        'sell_investment_amount': sell_amt,
        'investment_fraction': inv_frac,
        'use_limit_orders': limit,
        'buy_threshold': buy_thr,
        'sell_threshold': sell_thr,
        'limit_order_buffer': lob,
        'limit_order_duration': lod,
        'stop_loss_pct': slp,
        'freeze_window': fw,
        'Total Return (%)': strategy_metrics['Total Return'],
        'Annualized Return (%)': strategy_metrics['Annualized Return'],
        'Sharpe Ratio': strategy_metrics['Annualized Sharpe Ratio'],
        'Maximum Drawdown (%)': strategy_metrics['Maximum Drawdown'],
        'Number of Transactions': strategy_metrics['Number of Transactions']
    }
    return result
