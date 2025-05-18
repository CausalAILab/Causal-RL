import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
from io import StringIO

# Smoothing function for reward curves
def smooth(scalars: list[float], weight: float) -> list[float]:  # Weight between 0 and 1 (e.g., 0.9)
    if not scalars:
        return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def load_experiment_data(experiment_log_dir: str) -> pd.DataFrame | None:
    """
    Loads and preprocesses monitor data from all .monitor.csv files
    for a single experiment configuration.
    """
    all_monitor_files = glob.glob(os.path.join(experiment_log_dir, "*.monitor.csv"))
    if not all_monitor_files:
        # Try looking in eval_logs subdirectory as well if the primary is empty (e.g. if only eval was run)
        all_monitor_files = glob.glob(os.path.join(experiment_log_dir, "eval_logs", "*.monitor.csv"))
        if not all_monitor_files:
            print(f"No monitor files found in {experiment_log_dir} or its eval_logs subdir.")
            return None

    df_list = []
    print(f"Loading data for experiment: {os.path.basename(experiment_log_dir)}")
    for i, f_path in enumerate(all_monitor_files):
        try:
            with open(f_path, 'r') as f:
                lines = f.readlines()
            
            header_comment_lines_count = 0
            # Skip initial comment lines (which usually contain JSON for monitor.csv)
            while header_comment_lines_count < len(lines) and lines[header_comment_lines_count].strip().startswith('#'):
                header_comment_lines_count += 1
            
            if header_comment_lines_count < len(lines):
                # csv_content now starts from the first non-comment line (expected to be the CSV header)
                csv_content = "".join(lines[header_comment_lines_count:])
                # Check if csv_content is empty or just whitespace
                if not csv_content.strip():
                    # print(f"Skipping {f_path}: effective content is empty after stripping JSON/comment header.")
                    continue
                
                temp_df = pd.read_csv(StringIO(csv_content))
                
                # Extract process_id from filename, e.g., "0.monitor.csv" -> "0"
                filename_base = os.path.basename(f_path)
                process_id_str = filename_base.split('.')[0] 
                # Attempt to convert to int if it's numeric, otherwise use string.
                try:
                    temp_df['process_id'] = int(process_id_str)
                except ValueError:
                    temp_df['process_id'] = process_id_str

                required_cols = {'r', 'l', 't'}
                if not required_cols.issubset(temp_df.columns):
                    print(f"Skipping {f_path}: missing standard columns (r, l, t) after parsing. Found: {temp_df.columns.tolist()}")
                    continue
                
                if 'door_open_state' not in temp_df.columns:
                    # print(f"Warning: 'door_open_state' not found in {f_path} for {os.path.basename(experiment_log_dir)}. Filling with -1.")
                    temp_df['door_open_state'] = -1.0 
                
                if 'active_dofs' not in temp_df.columns:
                    # print(f"Warning: 'active_dofs' not found in {f_path} for {os.path.basename(experiment_log_dir)}.")
                    # Try to infer from name or set to NaN, for now, let it be missing if not critical for a plot
                    pass

                df_list.append(temp_df)
            # else:
                # print(f"Skipping {f_path}: no data lines found after JSON/comment header.")

        except pd.errors.EmptyDataError:
            # print(f"Skipping empty monitor file: {f_path}")
            pass
        except Exception as e:
            print(f"Error reading {f_path}: {e}")
            
    if not df_list:
        print(f"No valid data loaded from any monitor file in {experiment_log_dir}")
        return None

    full_df = pd.concat(df_list, ignore_index=True)
    
    # Sort by time primarily to make cumsum deterministic if logs from different processes overlap in 't'
    full_df = full_df.sort_values(by='t').reset_index(drop=True)
    full_df['total_timesteps'] = full_df['l'].cumsum()
    
    # Convert door_open_state to boolean for easier aggregation (1.0 is open)
    # full_df['door_opened'] = full_df['door_open_state'] == 1.0
    full_df['door_opened'] = full_df['r'] > 0.0
    
    return full_df

def plot_average_rewards(experiments_data: dict[str, pd.DataFrame], output_dir: str, exp_type: str, smoothing_weight=0.9):
    """Plots smoothed average episode rewards vs. total timesteps."""
    plt.figure(figsize=(15, 8))
    for exp_name, df in experiments_data.items():
        if df is not None and 'r' in df.columns and 'total_timesteps' in df.columns:
            rewards = df['r'].tolist()
            timesteps = df['total_timesteps'].tolist()
            if rewards and timesteps:
                smoothed_rewards = smooth(rewards, smoothing_weight)
                plt.plot(timesteps, smoothed_rewards, label=exp_name, alpha=0.7)
    
    plt.xlabel("Total Timesteps")
    plt.ylabel(f"Smoothed Average Episode Reward (Weight: {smoothing_weight})")
    plt.title(f"Average Reward vs. Timesteps for '{exp_type.upper()}' Experiments")
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{exp_type}_average_rewards.png")
    plt.savefig(output_path)
    print(f"Saved average rewards plot to {output_path}")
    plt.close()

def plot_door_open_success_rate(experiments_data: dict[str, pd.DataFrame], output_dir: str, exp_type: str):
    """Plots the percentage of episodes ending with the door open."""
    exp_names = []
    success_rates = []

    for exp_name, df in experiments_data.items():
        if df is not None and 'door_opened' in df.columns and not df.empty:
            rate = df['door_opened'].mean() * 100
            exp_names.append(exp_name)
            success_rates.append(rate)
        elif df is not None and df.empty:
             exp_names.append(exp_name)
             success_rates.append(0)


    if not exp_names:
        print("No data to plot for door open success rate.")
        return

    plt.figure(figsize=(max(10, len(exp_names) * 0.5), 6)) # Adjust width based on num_exps
    bars = plt.bar(exp_names, success_rates)
    plt.xlabel("Experiment Configuration")
    # plt.ylabel("Door Open Success Rate at Episode End (%)")
    # plt.title(f"Door Open Success Rate for '{exp_type.upper()}' Experiments")
    plt.ylabel("Door Open Success Rate (Episode Reward > 0) (%)")
    plt.title(f"Door Open Success Rate (Reward-Based) for '{exp_type.upper()}' Experiments")
    plt.xticks(rotation=45, ha="right", fontsize='small')
    plt.yticks(np.arange(0, 101, 10))
    plt.grid(axis='y', linestyle='--')
    # Add text labels on bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize='x-small')
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{exp_type}_door_open_success_rate.png")
    plt.savefig(output_path)
    print(f"Saved door open success rate plot to {output_path}")
    plt.close()

def plot_cumulative_wins(experiments_data: dict[str, pd.DataFrame], output_dir: str, exp_type: str):
    """Plots cumulative count of successful episodes vs. total timesteps."""
    plt.figure(figsize=(15, 8))
    for exp_name, df in experiments_data.items():
        if df is not None and 'door_opened' in df.columns and 'total_timesteps' in df.columns:
            cumulative_wins = df['door_opened'].cumsum()
            timesteps = df['total_timesteps']
            if not cumulative_wins.empty and not timesteps.empty:
                 plt.plot(timesteps, cumulative_wins, label=exp_name, alpha=0.7)
    
    plt.xlabel("Total Timesteps")
    # plt.ylabel("Cumulative Successful Episodes (Door Open at End)")
    # plt.title(f"Cumulative Wins vs. Timesteps for '{exp_type.upper()}' Experiments")
    plt.ylabel("Cumulative Successful Episodes (Episode Reward > 0)")
    plt.title(f"Cumulative Wins (Reward-Based) vs. Timesteps for '{exp_type.upper()}' Experiments")
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{exp_type}_cumulative_wins.png")
    plt.savefig(output_path)
    print(f"Saved cumulative wins plot to {output_path}")
    plt.close()

def plot_average_rewards_with_variance(
    experiments_data: dict[str, pd.DataFrame], 
    output_dir: str, 
    exp_type: str, 
    smoothing_weight=0.95, # Match default from args
    num_bins=50 
):
    plt.figure(figsize=(15, 8))
    
    for exp_name, df_config in experiments_data.items():
        if df_config is None or df_config.empty or \
           'total_timesteps' not in df_config.columns or \
           'process_id' not in df_config.columns:
            print(f"Skipping variance plot for {exp_name}: data missing columns ('total_timesteps', 'process_id') or empty.")
            continue

        min_ts = df_config['total_timesteps'].min()
        max_ts = df_config['total_timesteps'].max()

        if pd.isna(min_ts) or pd.isna(max_ts):
             print(f"Skipping variance plot for {exp_name}: min/max total_timesteps are NaN.")
             continue
        
        if min_ts == max_ts:
            # Handle cases with very little data / all data at one timestep
            if num_bins <= 1:
                 bins = np.array([min_ts, max_ts + 1]) # Ensure one bin [min_ts, max_ts+1)
            else:
                 # Create multiple bins even if they are narrow, last one is adjusted
                 bins = np.linspace(min_ts, max_ts, num_bins + 1)
                 if bins[-1] <= bins[0]: # If all bins are min_ts (or max_ts)
                     bins[-1] = bins[0] + 1 # Make the last bin edge slightly larger
        else:
            bins = np.linspace(min_ts, max_ts, num_bins + 1)

        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        binned_means_of_process_means = []
        binned_stds_of_process_means = []
        
        for i in range(len(bin_centers)):
            bin_start, bin_end = bins[i], bins[i+1]
            
            # Ensure episodes ending exactly at bin_start are included in the first bin appropriately.
            # And episodes ending at bin_end are in *this* bin, not the next (left-closed, right-closed interval for episodes ending *within*).
            # For np.linspace, the last bin_end is max_ts. We want to include episodes ending at max_ts.
            if i == 0: # First bin includes start
                mask = (df_config['total_timesteps'] >= bin_start) & (df_config['total_timesteps'] <= bin_end)
            else: # Subsequent bins: (previous_bin_end, current_bin_end]
                mask = (df_config['total_timesteps'] > bin_start) & (df_config['total_timesteps'] <= bin_end)
            
            # If it's the very last bin, ensure it captures up to max_ts inclusively, even with float precision.
            if i == len(bin_centers) -1:
                 mask = (df_config['total_timesteps'] > bin_start) & (df_config['total_timesteps'] <= max_ts)

            episodes_in_bin_df = df_config[mask]
            
            if not episodes_in_bin_df.empty:
                per_process_avg_rewards = episodes_in_bin_df.groupby('process_id')['r'].mean()
                if not per_process_avg_rewards.empty:
                    mean_val = per_process_avg_rewards.mean()
                    std_val = per_process_avg_rewards.std(ddof=0) # std of means; ddof=0 for 1 process data = 0 std
                    binned_means_of_process_means.append(mean_val)
                    binned_stds_of_process_means.append(std_val if not pd.isna(std_val) else 0.0)
                else:
                    binned_means_of_process_means.append(np.nan)
                    binned_stds_of_process_means.append(np.nan)
            else:
                binned_means_of_process_means.append(np.nan)
                binned_stds_of_process_means.append(np.nan)

        means_np = np.array(binned_means_of_process_means)
        stds_np = np.array(binned_stds_of_process_means) # Already handled NaNs by converting to 0 for std.

        valid_indices = ~np.isnan(means_np)
        if not np.any(valid_indices):
            print(f"No valid binned data for variance plot of {exp_name}, skipping line.")
            continue
            
        plot_timesteps = bin_centers[valid_indices]
        plot_means = means_np[valid_indices]
        plot_stds = stds_np[valid_indices]

        if len(plot_means) == 0:
            continue
        elif len(plot_means) < 2:
            smoothed_plot_means_np = plot_means # No smoothing for single point
        else:
            smoothed_plot_means_np = np.array(smooth(plot_means.tolist(), smoothing_weight))
        
        line, = plt.plot(plot_timesteps, smoothed_plot_means_np, label=exp_name, alpha=0.9, linewidth=1.5)
        plt.fill_between(
            plot_timesteps,
            smoothed_plot_means_np - plot_stds, 
            smoothed_plot_means_np + plot_stds,
            color=line.get_color(),
            alpha=0.2
        )
        
    plt.xlabel(f"Total Timesteps (Binned into {num_bins} bins)")
    plt.ylabel(f"Smoothed Avg. Per-Process Reward (Weight: {smoothing_weight})")
    plt.title(f"Avg. Per-Process Reward & Std Dev across Processes for '{exp_type.upper()}' Experiments")
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{exp_type}_avg_rewards_process_variance.png")
    plt.savefig(output_path)
    print(f"Saved avg rewards with process variance plot to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot experiment results from Adroit Door PPO training. Can plot all configs of an experiment type or compare two specific experiment configs.')
    
    # Group for plotting a single experiment type (original behavior)
    group_single_type = parser.add_argument_group('Single Experiment Type Plotting')
    group_single_type.add_argument('--base-log-dir', type=str, default='logs/',
                        help='Base directory where experiment logs are stored (e.g., logs/). Used for single experiment type plotting.')
    group_single_type.add_argument('--experiment-type', type=str, choices=['pomis', 'random'],
                        help='Type of experiments to plot (pomis or random). If provided, all configs of this type are plotted.')

    # Group for comparing two specific experiment configurations
    group_comparison = parser.add_argument_group('Two Experiment Comparison Plotting')
    group_comparison.add_argument('--exp-A-name', type=str, help='Name for Experiment A (e.g., POMIS_Baseline) for file naming and default label.')
    group_comparison.add_argument('--exp-A-dir', type=str, help='Path to log directory for Experiment A')
    group_comparison.add_argument('--exp-B-name', type=str, help='Name for Experiment B (e.g., Random_AllActive) for file naming and default label.')
    group_comparison.add_argument('--exp-B-dir', type=str, help='Path to log directory for Experiment B')
    group_comparison.add_argument('--label-A', type=str, help='Custom legend/plot label for Experiment A. Overrides --exp-A-name for display.')
    group_comparison.add_argument('--label-B', type=str, help='Custom legend/plot label for Experiment B. Overrides --exp-B-name for display.')

    parser.add_argument('--output-dir', type=str, default='plots/',
                        help='Directory to save the generated plots')
    parser.add_argument('--smoothing-weight', type=float, default=0.95,
                        help='Smoothing weight for reward curves (0.0 to 1.0). Higher is smoother.')
    
    args = parser.parse_args()

    # Validate arguments: EITHER experiment-type OR (exp-A-dir AND exp-B-dir) must be provided
    if args.experiment_type and (args.exp_A_dir or args.exp_B_dir):
        parser.error("Cannot use --experiment-type with --exp-A-dir/--exp-B-dir. Choose one mode.")
    if (args.exp_A_dir and not args.exp_B_dir) or (not args.exp_A_dir and args.exp_B_dir):
        parser.error("--exp-A-dir and --exp-B-dir must be provided together for comparison.")
    if (args.exp_A_name and not args.exp_A_dir) or (not args.exp_A_name and args.exp_A_dir):
        parser.error("--exp-A-name must be provided with --exp-A-dir.")
    if (args.exp_B_name and not args.exp_B_dir) or (not args.exp_B_name and args.exp_B_dir):
        parser.error("--exp-B-name must be provided with --exp-B-dir.")
    if not args.experiment_type and not (args.exp_A_dir and args.exp_B_dir):
        parser.error("Either --experiment-type or both --exp-A-dir and --exp-B-dir must be specified.")

    os.makedirs(args.output_dir, exist_ok=True)
    experiment_configs_data = {}
    plot_title_exp_type = ""
    final_output_dir = args.output_dir

    if args.exp_A_dir and args.exp_B_dir: # Comparison mode
        print(f"\nStarting comparison between:")
        print(f"  Experiment A ({args.exp_A_name}): {args.exp_A_dir}")
        print(f"  Experiment B ({args.exp_B_name}): {args.exp_B_dir}")

        if not os.path.isdir(args.exp_A_dir):
            print(f"Error: Experiment A directory not found: {args.exp_A_dir}")
            return
        if not os.path.isdir(args.exp_B_dir):
            print(f"Error: Experiment B directory not found: {args.exp_B_dir}")
            return

        data_A = load_experiment_data(args.exp_A_dir)
        data_B = load_experiment_data(args.exp_B_dir)

        name_A_internal = args.exp_A_name if args.exp_A_name else os.path.basename(args.exp_A_dir)
        name_B_internal = args.exp_B_name if args.exp_B_name else os.path.basename(args.exp_B_dir)

        display_label_A = args.label_A if args.label_A else name_A_internal
        display_label_B = args.label_B if args.label_B else name_B_internal

        if data_A is not None:
            experiment_configs_data[display_label_A] = data_A
        else:
            print(f"Could not load data for Experiment A ({name_A_internal}) from {args.exp_A_dir}")
            experiment_configs_data[display_label_A] = pd.DataFrame()
        
        if data_B is not None:
            experiment_configs_data[display_label_B] = data_B
        else:
            print(f"Could not load data for Experiment B ({name_B_internal}) from {args.exp_B_dir}")
            experiment_configs_data[display_label_B] = pd.DataFrame()

        if not experiment_configs_data[display_label_A].empty or not experiment_configs_data[display_label_B].empty:
            plot_title_exp_type = f"Comparison: {display_label_A} vs {display_label_B}"
            # Sanitize display_label_A and display_label_B if they are used for directory names to avoid issues with special characters
            # For simplicity, we'll assume they are reasonably filesystem-friendly or rely on user to provide good names.
            # A more robust solution might involve slugifying them.
            comparison_folder_name = f"{display_label_A}_vs_{display_label_B}".replace(' ', '_').replace('/', '-') # Basic sanitization
            final_output_dir = os.path.join(args.output_dir, "compare", comparison_folder_name)
            os.makedirs(final_output_dir, exist_ok=True)
            print(f"Comparison plots will be saved to: {final_output_dir}")
        else:
            print("No data loaded for either experiment. Cannot generate comparison plots.")
            return

    elif args.experiment_type: # Single experiment type mode (original behavior)
        exp_type_path = os.path.join(args.base_log_dir, args.experiment_type)
        if not os.path.isdir(exp_type_path):
            print(f"Experiment type directory not found: {exp_type_path}")
            return
        
        print(f"\nLoading data for all configurations in {exp_type_path}...")
        for config_name in sorted(os.listdir(exp_type_path)):
            config_log_dir = os.path.join(exp_type_path, config_name)
            if os.path.isdir(config_log_dir):
                data_df = load_experiment_data(config_log_dir)
                if data_df is not None:
                    experiment_configs_data[config_name] = data_df
                else:
                    print(f"Could not load data for config: {config_name}")
                    experiment_configs_data[config_name] = pd.DataFrame()
        
        if not any(not df.empty for df in experiment_configs_data.values()):
            print(f"No data loaded for any experiment configuration under {exp_type_path}.")
            return
        plot_title_exp_type = args.experiment_type.upper()
        # final_output_dir is already args.output_dir for this case

    if not experiment_configs_data or not any(not df.empty for df in experiment_configs_data.values()):
        print("No data available to plot after loading attempts.")
        return

    print(f"\nPlotting results for {plot_title_exp_type}...")
    plot_average_rewards(experiment_configs_data, final_output_dir, plot_title_exp_type, args.smoothing_weight)
    plot_door_open_success_rate(experiment_configs_data, final_output_dir, plot_title_exp_type)
    plot_cumulative_wins(experiment_configs_data, final_output_dir, plot_title_exp_type)
    plot_average_rewards_with_variance(experiment_configs_data, final_output_dir, plot_title_exp_type, args.smoothing_weight)
    
    print(f"\nAll plots saved to {final_output_dir}")

if __name__ == '__main__':
    main() 