from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from itertools import product
from tqdm import tqdm
import pandas as pd
from runattempt import *  # Adjust if needed

if __name__ == "__main__":
    # Define parameters
    buy_investment_amount = [None]
    sell_investment_amount = [2_500_000]
    investment_fraction = [1.0]
    use_limit_orders = [True]
    buy_threshold = [0.655]
    sell_threshold = [0.31]
    limit_order_buffer = [0.00001]
    limit_order_duration = [10]
    stop_loss_pct = [0.04]
    freeze_window = [120]

    # Create all combinations of hyperparameters
    hyperparameter_combinations = list(product(
        buy_investment_amount,
        sell_investment_amount,
        investment_fraction,
        use_limit_orders,
        buy_threshold,
        sell_threshold,
        limit_order_buffer,
        limit_order_duration,
        stop_loss_pct,
        freeze_window
    ))

    print(f"Total combinations to evaluate: {len(hyperparameter_combinations)}")

    # Load your trading DataFrame
    trading_df = pd.read_pickle("full_trading.pkl")

    # Create partial function to include trading_df
    run_sim_with_df = partial(run_simulation, trading_df=trading_df)

    results = []
    total_tasks = len(hyperparameter_combinations)

    # Run parallel simulations
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_sim_with_df, params) for params in hyperparameter_combinations]

        # Use tqdm to show a progress bar
        for future in tqdm(as_completed(futures), total=total_tasks, desc="Processing"):
            res = future.result()
            if res is not None:
                results.append(res)

    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)
    results_df.to_csv("hyper_results.csv")
    # Convert percentage strings to floats for sorting if necessary
    results_df['Total Return (%)'] = results_df['Total Return (%)'].str.rstrip('%').astype('float') / 100.0
    results_df['Annualized Return (%)'] = results_df['Annualized Return (%)'].str.rstrip('%').astype('float') / 100.0
    results_df['Maximum Drawdown (%)'] = results_df['Maximum Drawdown (%)'].str.rstrip('%').astype('float') / 100.0
    results_df['Sharpe Ratio'] = results_df['Sharpe Ratio'].astype('float')

    # Find the best performing strategy based on Sharpe Ratio
    best_strategy = results_df.loc[results_df['Sharpe Ratio'].idxmax()]

    print("\nBest Strategy Parameters:")
    print(best_strategy)

    # Display the top N strategies
    top_n = 5
    print(f"\nTop {top_n} Strategies:")
    print(results_df.sort_values(by='Total Return (%)', ascending=False).head(top_n))

    # Save results to CSV
    results_df.to_csv("hyper_results.csv")
