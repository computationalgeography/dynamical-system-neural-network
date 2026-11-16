import pandas as pd
from pathlib import Path

# folder containing hyper parameter tuning results
output_folder_lstm = "/Users/karss101/misc/sync/misc/research/models/dyn_sys_nn/figures/lstm_hyper_parameter_tuning/"

# folders containing hyper parameter tuning results within
# that folder
folders = ['247', '35',  '528', '534', '535', '565', '68',  '815', '818']


# Create an empty DataFrame
result = pd.DataFrame()

# Read each CSV file and select the 'test' column
for folder in folders:
    file_name = output_folder_lstm + folder + '/hyperparameter_tuning_results.csv'

    df = pd.read_csv(file_name)

    # Add the 'test' column to the result DataFrame
    result[folder] = df["best_validation_rmse_scaled"]

result["median"] = result.median(axis=1)
result["is_min"] = result["median"] == result["median"].min()

# Display the resulting DataFrame
print(result)

