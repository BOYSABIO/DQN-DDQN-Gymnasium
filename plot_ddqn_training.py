import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

def plot_training_progress(csv_path, window_size=50, save_path='ddqn_training_progress.png'):
    # Load CSV
    df = pd.read_csv(csv_path, header=None)

    # Extract episode times (skip label "time")
    times = df.iloc[1, 1:].values
    time_fmt = "%H:%M:%S"
    start_time = datetime.strptime(times[0], time_fmt)
    end_time = datetime.strptime(times[-1], time_fmt)
    if end_time < start_time:  # handle wrap-around midnight
        end_time = end_time.replace(day=start_time.day + 1)
    duration = end_time - start_time

    # Extract reward values (skip label "reward")
    rewards_str = df.iloc[2, 1:].values
    rewards = pd.to_numeric(rewards_str, errors='coerce')
    rewards_series = pd.Series(rewards)

    # Compute sliding window average
    rolling_avg = rewards_series.rolling(window=window_size, min_periods=1).mean()

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(rewards_series, label="Episode Reward", alpha=0.4)
    plt.plot(rolling_avg, label=f"Moving Average (window={window_size})", linewidth=2, color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"DDQN Training Progress\nTotal Training Time: {duration}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot DDQN training progress from log CSV.")
    parser.add_argument('--csv', type=str, required=True, help='Path to DDQN log CSV file (e.g., ./logs/DDQN_final_log.csv)')
    parser.add_argument('--window', type=int, default=50, help='Moving average window size (default: 50)')
    parser.add_argument('--output', type=str, default='ddqn_training_progress.png', help='Output plot filename (default: ddqn_training_progress.png)')
    args = parser.parse_args()

    plot_training_progress(args.csv, window_size=args.window, save_path=args.output) 