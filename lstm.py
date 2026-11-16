import copy
import numpy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
import sys
import itertools

# LSTM model to create benchmark for DSNN
# Derek Karssenberg, 16 Nov 2026

# ================
# 1. Inputs, setup
# ================

catchment_id = sys.argv[1]

data_folder = (
    "../data/results_all_catch/land_obs_two/" + catchment_id + "/fit_eva/1/1/arrays/"
)

# output_folder_lstm = "../figures/lstm_hyper_parameter_tuning/"
output_folder_lstm = "../figures/lstm_predictions/"

out_folder = output_folder_lstm + catchment_id
out_folder_p = out_folder + "/"

try:
    os.mkdir(out_folder)
    print(f"Directory '{out_folder}' created successfully.")
except FileExistsError:
    print(f"Directory '{out_folder}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{out_folder}'.")
except Exception as e:
    print(f"An error occurred: {e}")

# training data

# features, inputs
temp = data_folder + "train_ts_temperature.npy"
prec = data_folder + "train_ts_precipitation.npy"
# response variable, streamflow
flow = data_folder + "train_ts_OBS.npy"

# testing (independent validation) data
# features, inputs
test_temp = data_folder + "valid_ts_temperature.npy"
test_prec = data_folder + "valid_ts_precipitation.npy"
# response variable, streamflow
test_flow = data_folder + "valid_ts_OBS.npy"

# proportion used for training
train_ratio = 0.70

# window size in days, this will also
# ignore this time span in validation, which was
# 365 in DSNN (spin up), so use it here as well
window_size = 365

# hyperparameter tuning candidates
# for hyperparameter tuning
# hidden_size_options = [32, 64, 128]
# learning_rate_options = [0.0001, 0.001, 0.003]
# batch_size_options = [32, 64, 128]
# for inference (testing, independent validation)
hidden_size_options = [64]
learning_rate_options = [0.003]
batch_size_options = [64]

num_layers = 2
dropout = 0.2
epochs = 200
patience = 20

seed = 42
torch.manual_seed(seed)
numpy.random.seed(seed)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Running catchment id: {catchment_id}")
print(f"Using device: {device}")
print(f"Input window: {window_size} days")
print("Hyperparameter tuning candidates:")
print(f"  Hidden sizes: {hidden_size_options}")
print(f"  Learning rates: {learning_rate_options}")
print(f"  Batch sizes: {batch_size_options}")
print(
    f"  Total configurations: {len(hidden_size_options) * len(learning_rate_options) * len(batch_size_options)}"
)

# =======================================
# 2. Load training/validation time series
# =======================================

x1 = numpy.load(temp).reshape(-1)
x2 = numpy.load(prec).reshape(-1)
y = numpy.load(flow).reshape(-1)

if not (len(x1) == len(x2) == len(y)):
    raise ValueError(
        "Training input files and output file must have the same number of observations."
    )

X = numpy.column_stack((x1, x2)).astype(numpy.float32)
y = y.astype(numpy.float32)

if not numpy.isfinite(X).all() or not numpy.isfinite(y).all():
    raise ValueError("Training data contain NaN or infinite values.")

# set index for splitting in training and validation
n = len(y)
split_index = int(train_ratio * n)

if split_index <= window_size or split_index >= n:
    raise ValueError(
        f"Need more than {window_size} training observations and a non-empty "
        "validation period. Check the data and train_ratio."
    )

print("\nTraining/validation data")
print(f"Total observations: {n}")
print(f"Training target period: indices {window_size} to {split_index - 1}")
print(f"Validation target period: indices {split_index} to {n - 1}")
print(f"First {window_size} observations are used as initial history, not targets.")

# =============================
# Load independent testing data
# =============================

test_x1 = numpy.load(test_temp).reshape(-1)
test_x2 = numpy.load(test_prec).reshape(-1)
test_y = numpy.load(test_flow).reshape(-1).astype(numpy.float32)

if not (len(test_x1) == len(test_x2) == len(test_y)):
    raise ValueError(
        "Testing input files and testing output file must have the same number of observations."
    )

X_test = numpy.column_stack((test_x1, test_x2)).astype(numpy.float32)

if len(test_y) <= window_size:
    raise ValueError(
        f"The independent test dataset must contain more than {window_size} observations "
        "to produce at least one prediction with a 365-day input window."
    )
if not numpy.isfinite(X_test).all() or not numpy.isfinite(test_y).all():
    raise ValueError("Testing data contain NaN or infinite values.")

print("\nIndependent testing data")
print(f"Testing observations: {len(test_y)}")
print(
    f"Test predictions will be produced for indices {window_size} to {len(test_y) - 1}."
)
print(f"The first {window_size} test days provide history for the first prediction.")

# ===============================
# Standardize using training data
# ===============================

x_mean = X[:split_index].mean(axis=0)
x_std = X[:split_index].std(axis=0)
# prevent zero std (will not happen here)
x_std[x_std < 1e-8] = 1.0

y_mean = y[:split_index].mean()
y_std = y[:split_index].std()
# prevent zero std
if y_std < 1e-8:
    y_std = 1.0

X_scaled = ((X - x_mean) / x_std).astype(numpy.float32)
y_scaled = ((y - y_mean) / y_std).astype(numpy.float32)
X_test_scaled = ((X_test - x_mean) / x_std).astype(numpy.float32)

# ======================
# Create sliding windows
# ======================


def make_windows(X_values, y_values, target_start, target_end, window_size):
    """Make windows predicting y[t] from X[t-window_size:t].
    target_start is inclusive; target_end is exclusive. The window contains
    the 365 input days immediately BEFORE its target day (not the target day).
    """
    first_target = max(window_size, target_start)
    target_indices = numpy.arange(first_target, target_end, dtype=numpy.int64)

    if len(target_indices) == 0:
        return (
            numpy.empty((0, window_size, X_values.shape[1]), dtype=numpy.float32),
            numpy.empty((0, 1), dtype=numpy.float32),
            target_indices,
        )

    X_windows = numpy.stack(
        [X_values[t - window_size : t] for t in target_indices]
    ).astype(numpy.float32)
    y_targets = y_values[target_indices].reshape(-1, 1).astype(numpy.float32)
    return X_windows, y_targets, target_indices


# Training targets are before the chronological split
X_train_windows, y_train_targets, train_target_indices = make_windows(
    X_scaled, y_scaled, window_size, split_index, window_size
)

# Validation targets start at the split. Their input windows may include the
# preceding training-period input history, which is available at prediction time
X_val_windows, y_val_targets, val_target_indices = make_windows(
    X_scaled, y_scaled, split_index, n, window_size
)

if len(X_train_windows) == 0 or len(X_val_windows) == 0:
    raise ValueError(
        "Not enough observations to create training and validation windows."
    )

train_dataset = TensorDataset(
    torch.from_numpy(X_train_windows), torch.from_numpy(y_train_targets)
)
val_dataset = TensorDataset(
    torch.from_numpy(X_val_windows), torch.from_numpy(y_val_targets)
)

print("\nWindow datasets")
print(f"Training windows: {len(train_dataset)}")
print(f"Validation windows: {len(val_dataset)}")
print(f"Window tensor shape: ({window_size}, 2)")
exit()

# ==============================
# Define LSTM and tuning helpers
# ==============================


class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, .. days, 2 features)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # one prediction for the next day


def rmse_loss(prediction, target):
    """Calculate RMSE loss"""
    return torch.sqrt(torch.mean((prediction - target) ** 2) + 1e-8)


def evaluate_loader(model, loader):
    """Calculate RMSE over all samples in standardized target units."""
    model.eval()
    squared_error_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            predictions = model(batch_X)
            squared_error_sum += torch.sum((predictions - batch_y) ** 2).item()
            sample_count += batch_y.numel()
    return float(numpy.sqrt(squared_error_sum / sample_count))


def train_one_configuration(hidden_size, learning_rate, batch_size, trial_number):
    """Train one candidate and return its best validation RMSE/model state."""
    # Reset the seed so configurations start from comparable initial conditions.
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    candidate_model = LSTMModel(
        input_size=2,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(candidate_model.parameters(), lr=learning_rate)

    best_candidate_val_rmse = float("inf")
    best_candidate_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    candidate_train_losses = []
    candidate_val_losses = []

    print(
        f"\\nTrial {trial_number:02d}: hidden_size={hidden_size}, "
        f"learning_rate={learning_rate:g}, batch_size={batch_size}"
    )

    for epoch in range(epochs):
        candidate_model.train()
        squared_error_sum = 0.0
        sample_count = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            batch_predictions = candidate_model(batch_X)
            loss = rmse_loss(batch_predictions, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(candidate_model.parameters(), max_norm=1.0)
            optimizer.step()

            squared_error_sum += torch.sum(
                (batch_predictions.detach() - batch_y) ** 2
            ).item()
            sample_count += batch_y.numel()

        epoch_train_rmse = float(numpy.sqrt(squared_error_sum / sample_count))
        epoch_val_rmse = evaluate_loader(candidate_model, val_loader)
        candidate_train_losses.append(epoch_train_rmse)
        candidate_val_losses.append(epoch_val_rmse)

        # stopping based on validation loss
        if epoch_val_rmse < best_candidate_val_rmse:
            best_candidate_val_rmse = epoch_val_rmse
            best_candidate_state = copy.deepcopy(candidate_model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_candidate_state is None:
        raise RuntimeError("A tuning trial did not produce a valid model state.")

    print(
        f"  Best validation RMSE (scaled): {best_candidate_val_rmse:.6f} "
        f"at epoch {best_epoch}"
    )
    return {
        "hidden_size": hidden_size,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "best_val_rmse_scaled": best_candidate_val_rmse,
        "best_epoch": best_epoch,
        "model_state_dict": copy.deepcopy(best_candidate_state),
        "train_losses": candidate_train_losses,
        "val_losses": candidate_val_losses,
    }


# ============================================================
# Hyperparameter grid search
# ============================================================

search_space = list(
    itertools.product(hidden_size_options, learning_rate_options, batch_size_options)
)
trial_results = []
best_result = None

for trial_number, (hidden_size, learning_rate, batch_size) in enumerate(
    search_space, start=1
):
    result = train_one_configuration(
        hidden_size, learning_rate, batch_size, trial_number
    )
    trial_results.append(result)
    if (
        best_result is None
        or result["best_val_rmse_scaled"] < best_result["best_val_rmse_scaled"]
    ):
        best_result = result

if best_result is None:
    raise RuntimeError("Hyperparameter tuning did not complete successfully.")

best_hidden_size = best_result["hidden_size"]
best_learning_rate = best_result["learning_rate"]
best_batch_size = best_result["batch_size"]
train_losses = best_result["train_losses"]
val_losses = best_result["val_losses"]

# Recreate the best model configuration and load its best validation checkpoint
model = LSTMModel(
    input_size=2,
    hidden_size=best_hidden_size,
    num_layers=num_layers,
    dropout=dropout,
).to(device)
model.load_state_dict(best_result["model_state_dict"])
model.eval()

best_hyperparameters_text = "\n".join(
    [
        "=" * 65,
        "BEST HYPERPARAMETERS (selected using validation RMSE)",
        f"Hidden size:                 {best_hidden_size}",
        f"Learning rate:               {best_learning_rate:g}",
        f"Batch size:                  {best_batch_size}",
        f"Best validation RMSE scaled: {best_result['best_val_rmse_scaled']:.6f}",
        f"Best epoch:                  {best_result['best_epoch']}",
        "=" * 65,
    ]
)
print("\n" + best_hyperparameters_text)

with open(out_folder_p + "best_hyperparameters.txt", "w", encoding="utf-8") as txt_file:
    txt_file.write(best_hyperparameters_text + "\n")
print("Saved best hyperparameters to best_hyperparameters.txt")

# Save every trial's results
tuning_summary = numpy.array(
    [
        [
            r["hidden_size"],
            r["learning_rate"],
            r["batch_size"],
            r["best_val_rmse_scaled"],
            r["best_epoch"],
        ]
        for r in trial_results
    ],
    dtype=numpy.float64,
)
numpy.savetxt(
    out_folder_p + "hyperparameter_tuning_results.csv",
    tuning_summary,
    delimiter=",",
    header="hidden_size,learning_rate,batch_size,best_validation_rmse_scaled,best_epoch",
    comments="",
    fmt=["%d", "%.8g", "%d", "%.8f", "%d"],
)

# ==========================================================
# Batch prediction helper and training/validation evaluation
# ==========================================================


def predict_windows(X_windows, batch_size=best_batch_size):
    """Predict windows in batches and return predictions in scaled units."""
    if len(X_windows) == 0:
        return numpy.empty((0,), dtype=numpy.float32)
    dataset = TensorDataset(
        torch.from_numpy(numpy.asarray(X_windows, dtype=numpy.float32))
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    output_batches = []
    model.eval()
    with torch.no_grad():
        for (batch_X,) in loader:
            batch_predictions = model(batch_X.to(device))
            output_batches.append(batch_predictions.cpu().numpy().reshape(-1))
    return numpy.concatenate(output_batches)


train_pred_scaled = predict_windows(X_train_windows)
val_pred_scaled = predict_windows(X_val_windows)

train_pred = train_pred_scaled * y_std + y_mean
val_pred = val_pred_scaled * y_std + y_mean
train_observed = y[train_target_indices]
val_observed = y[val_target_indices]

train_rmse_original = float(numpy.sqrt(numpy.mean((train_pred - train_observed) ** 2)))
val_rmse_original = float(numpy.sqrt(numpy.mean((val_pred - val_observed) ** 2)))

print("\nTraining and validation RMSE (original units)")
print(f"Training RMSE:   {train_rmse_original:.6f}")
print(f"Validation RMSE: {val_rmse_original:.6f}")

# Store predictions in full-length arrays; the first 365 entries have no prediction.
all_predictions = numpy.full(n, numpy.nan, dtype=numpy.float64)
all_predictions[train_target_indices] = train_pred
all_predictions[val_target_indices] = val_pred

# ===================
# Independent testing
# ===================

# The independent test sequence is windowed on its own. Its first window_size days
# are used as input history; the first test prediction is for test index window_size
X_test_windows, test_y_targets_scaled, test_target_indices = make_windows(
    X_test_scaled,
    ((test_y - y_mean) / y_std).astype(numpy.float32),
    window_size,
    len(test_y),
    window_size,
)

test_predictions_scaled = predict_windows(X_test_windows)
test_predictions = test_predictions_scaled * y_std + y_mean
test_observed = test_y[test_target_indices]

if len(test_observed) == 0:
    raise ValueError("No independent test predictions could be generated.")

test_rmse = float(numpy.sqrt(numpy.mean((test_predictions - test_observed) ** 2)))
test_mae = float(numpy.mean(numpy.abs(test_predictions - test_observed)))
test_mse = float(numpy.mean((test_predictions - test_observed) ** 2))

# Pearson correlation coefficient (r)
if numpy.std(test_observed) > 1e-8 and numpy.std(test_predictions) > 1e-8:
    test_correlation = float(numpy.corrcoef(test_observed, test_predictions)[0, 1])
else:
    test_correlation = float("nan")

# Percent Bias
observed_sum = numpy.sum(test_observed)
if abs(observed_sum) > 1e-8:
    pBias = float(numpy.sum(test_predictions - test_observed) / observed_sum * 100.0)
else:
    pBias = float("nan")

# Nash-Sutcliffe Efficiency (NSE)
sse = numpy.sum((test_observed - test_predictions) ** 2)
sst = numpy.sum((test_observed - numpy.mean(test_observed)) ** 2)
NSE = float(1.0 - sse / sst) if sst > 1e-8 else float("nan")

print("\nIndependent test results (original units)")
print(f"Test predictions evaluated: {len(test_observed)}")
print(f"Test RMSE:                   {test_rmse:.6f}")
print(f"Test MSE:                    {test_mse:.6f}")
print(f"Test MAE:                    {test_mae:.6f}")
print(f"Test pBias:                  {pBias:.4f}%")
print(f"Test NSE:                    {NSE:.6f}")
print(f"Test correlation (Pearson r): {test_correlation:.6f}")


metrics = numpy.array(
    [test_rmse, test_mse, test_mae, pBias, NSE, test_correlation]
).reshape(1, -1)
numpy.savetxt(
    out_folder_p + "metrics.csv",
    metrics,
    delimiter=",",
    header="rmse, mse, mae, pBias, NSE, correlation",
    comments="",
)

# ===============================
# Plot independent testing output
# ===============================

plt.figure(figsize=(14, 6))
plt.plot(
    test_target_indices,
    test_observed,
    label="Observed Output",
    color="blue",
    linewidth=1.5,
)
plt.plot(
    test_target_indices,
    test_predictions,
    label="LSTM Predicted Output",
    color="red",
    linewidth=1.5,
    linestyle="--",
)
plt.xlabel("Test time index (days)")
plt.ylabel("Output")
plt.title(f"Testing: Observed vs Predicted (RMSE={test_rmse:.4f})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_folder_p + "testing_observed_vs_predicted.png", dpi=300)
# plt.show()

# =====================
# Save prediction files
# =====================

train_results = numpy.column_stack((train_target_indices, train_observed, train_pred))
val_results = numpy.column_stack((val_target_indices, val_observed, val_pred))
test_results = numpy.column_stack(
    (test_target_indices, test_observed, test_predictions)
)

numpy.savetxt(
    out_folder_p + "training_predictions.txt",
    train_results,
    header="time_index observed predicted",
    fmt=["%d", "%.8f", "%.8f"],
)
numpy.savetxt(
    out_folder_p + "validation_predictions.txt",
    val_results,
    header="time_index observed predicted",
    fmt=["%d", "%.8f", "%.8f"],
)
numpy.savetxt(
    out_folder_p + "all_predictions.txt",
    numpy.column_stack((numpy.arange(n), y, all_predictions)),
    header="time_index observed predicted_nan_for_first_365_days",
    fmt=["%d", "%.8f", "%.8f"],
)
numpy.savetxt(
    out_folder_p + "testing_predictions.txt",
    test_results,
    header="test_time_index observed predicted",
    fmt=["%d", "%.8f", "%.8f"],
)

# ===============================================
# Save trained model and preprocessing parameters
# ===============================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "hidden_size": best_hidden_size,
        "num_layers": num_layers,
        "input_size": 2,
        "dropout": dropout,
        "window_size": window_size,
        "batch_size": best_batch_size,
        "learning_rate": best_learning_rate,
        "best_validation_rmse_scaled": best_result["best_val_rmse_scaled"],
    },
    out_folder_p + "lstm_model.pth",
)

# =====================
# Plot training history
# =====================

plt.figure(figsize=(10, 5))
plt.plot(numpy.arange(1, len(train_losses) + 1), train_losses, label="Training RMSE")
plt.plot(numpy.arange(1, len(val_losses) + 1), val_losses, label="Validation RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE (standardized output units)")
plt.title("Best LSTM Training History (365-day windows, mini-batches)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_folder_p + "training_history.png", dpi=150)
# plt.show()

# ============================================
# Plot observed vs predicted validation output
# ============================================

plt.figure(figsize=(12, 5))
plt.plot(
    val_target_indices, val_observed, label="Observed validation output", linewidth=1.5
)
plt.plot(
    val_target_indices, val_pred, label="Predicted validation output", linewidth=1.5
)
plt.axvline(split_index, linestyle="--", label="70/30 split")
plt.xlabel("Time index (days)")
plt.ylabel("Output")
plt.title(f"Validation: Observed vs Predicted (RMSE={val_rmse_original:.4f})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_folder_p + "validation_predictions.png", dpi=150)
# plt.show()

print(
    "\nFinished. Saved hyperparameter_tuning_results.csv, prediction files, plots, and lstm_model.pth to the current working directory."
)
