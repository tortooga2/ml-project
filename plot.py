import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Load Data (Modify 'your_results_file.csv' and 'your_score_column' name) ---
file_name = 'xgb_cross_validation_full_results.csv'
score_col = 'mean_test_score' # Assuming this is your mean score column
std_col = 'std_test_score'    # Assuming this is your standard deviation column

try:
    df = pd.read_csv(file_name)
except FileNotFoundError:
    print(f"Error: File '{file_name}' not found. Please check the file name.")
    # Create a dummy DataFrame for demonstration if the file is not available
    data = {
        'param_n_estimators': [50, 50, 100, 100, 200, 200],
        'param_max_depth': [3, 5, 3, 5, 3, 5],
        score_col: [0.90, 0.91, 0.93, 0.94, 0.95, 0.96],
        std_col: [0.01, 0.005, 0.008, 0.004, 0.003, 0.002]
    }
    df = pd.DataFrame(data)
    print("Using a dummy DataFrame for demonstration.")


# --- 2. Filter Results for Plotting ---
# When you have multiple hyperparameters, you must filter the results
# to isolate the trend for the parameter of interest (param_n_estimators).
# A common method is to select the results corresponding to the best values
# of the *other* parameters.

df[score_col] = 1 - df[score_col]

# Identify the best parameters (excluding the one you want to plot)
best_score = df[score_col].min()
best_row = df[df[score_col] == best_score].iloc[0]



# Hyperparameters to keep constant for the plot
fixed_params = ['param_learning_rate', 'param_max_depth']
# Create a filter based on the best values of the fixed parameters
filter_mask = pd.Series([True] * len(df))

for param in fixed_params:
    if param in df.columns and param != 'param_n_estimators':
        # Apply the filter only if the column exists and is not the X-axis parameter
        filter_mask &= (df[param] == best_row[param])

# Filtered DataFrame
plot_df = df[filter_mask].sort_values(by='param_n_estimators')

# --- 3. Generate the Matplotlib Plot ---
plt.figure(figsize=(10, 6))

# Plot the mean score with error bars (std_test_score)
plt.errorbar(
    x=plot_df['param_n_estimators'],
    y=plot_df[score_col],
    yerr=plot_df[std_col],
    label='Mean Test Score',
    marker='o',
    capsize=4
)

# Add title and labels
plt.title(f'Mean Error vs. Number of Estimators\n(Fixed Params: {best_row[fixed_params].to_dict()})')
plt.xlabel('param_n_estimators (Number of Trees)')
plt.ylabel('Cross-Validated Test Error')
plt.grid(True)
plt.legend()

# Save the plot
plt.savefig('Score_vs_Estimators_Replication.png')
print("Plot saved as 'Score_vs_Estimators_Replication.png'")