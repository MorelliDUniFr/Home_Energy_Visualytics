# Data manipulation
import random

import numpy as np
import pandas as pd
import itertools

# PyTorch core
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR

# Preprocessing
from sklearn.preprocessing import MinMaxScaler

import time

SEED = 42
pd.set_option("display.max_colwidth", None)

class SlidingWindowDataset(Dataset):
    def __init__(self, data, input_columns, target_columns, sequence_length, stride=1, time_threshold='10s'):
        self.sequence_length = sequence_length
        self.stride = int(sequence_length * stride)
        self.input_columns = input_columns
        self.target_columns = target_columns

        self.data = data.copy()
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])

        # Optional: One-hot encode household_id_column if categorical
        if 'household_id_column' in self.input_columns:
            self.data = pd.get_dummies(self.data, columns=['household_id_column'], drop_first=True)

        # Convert input/target columns to numeric
        self.data[self.input_columns] = self.data[self.input_columns].apply(pd.to_numeric, errors='coerce')
        self.data[self.target_columns] = self.data[self.target_columns].apply(pd.to_numeric, errors='coerce')

        # Handle multiple households if column exists
        household_col = 'household_id_column' if 'household_id_column' in data.columns else None
        self.indices = []

        if household_col:
            group_cols = [household_col]
        else:
            group_cols = []

        # Sort by household and timestamp
        self.data = self.data.sort_values(group_cols + ['timestamp']).reset_index(drop=True)

        # Calculate time difference
        if household_col:
            self.data['time_diff'] = self.data.groupby(household_col)['timestamp'].diff()
        else:
            self.data['time_diff'] = self.data['timestamp'].diff()

        # Flag new segments where the time gap exceeds the threshold
        threshold = pd.Timedelta(time_threshold)
        self.data['segment'] = (self.data['time_diff'] > threshold).cumsum()

        # Group by segments (and household if needed)
        groupby_cols = ['segment'] + ([household_col] if household_col else [])

        for _, group in self.data.groupby(groupby_cols):
            group = group.reset_index(drop=True)
            for start in range(0, len(group) - sequence_length + 1, self.stride):
                self.indices.append((group, start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        group, start = self.indices[idx]
        x = group[self.input_columns].iloc[start : start + self.sequence_length].values
        y = group[self.target_columns].iloc[start : start + self.sequence_length].values

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y


class NILMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_rate=0.1):
        super(NILMModel, self).__init__()

        self.hidden_size = hidden_size

        # 1D convolution before LSTM
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=input_size,
                                kernel_size=3, padding=1)  # keeps same shape

        # LSTM layer (can be bidirectional)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout_rate)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Activation
        self.relu = nn.ReLU()

        self.output_layer = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        # x: (batch, seq_len, input_features)
        x = x.permute(0, 2, 1)  # (batch, input_size, seq_len)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)  # back to (batch, seq_len, input_size)

        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*directions)

        # Dropout
        dropout_out = self.dropout(lstm_out)

        # Activation
        relu_out = self.relu(dropout_out)

        out = self.output_layer(relu_out)  # (batch, seq_len, 1)


        return out


def select_device(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Check for GPU availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    return device


def create_dataloader_for_appliance(dataset, input_columns, target_appliance, sequence_length, stride, batch_size, seed):
    sw_dataset = SlidingWindowDataset(dataset, input_columns, target_appliance, sequence_length=sequence_length, stride=stride)

    print('Dataset length:', len(sw_dataset), 'windows\n')

    total_len = len(sw_dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    # Split the dataset randomly
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        sw_dataset, [train_len, val_len, test_len], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, generator=generator)  # shuffle=True for training
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=generator)

    # Get one sample to determine input and output dimensions
    sample_X, sample_y = train_dataset[0]
    seq_len = sample_X.shape[0]
    input_size = sample_X.shape[1]  # Number of input features
    output_size = sample_y.shape[1]  # Number of target appliances

    return train_loader, val_loader, test_loader, seq_len, input_size, output_size


def train_nilm_model(train_loader, val_loader, input_size, output_size, device, hidden_size, max_epochs, patience, seed, num_layers):
    model = NILMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers).to(device)

    criterion = torch.nn.SmoothL1Loss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.9)

    # Initialize variables for tracking
    best_loss = float('inf')
    patience_counter = 0

    # To track loss
    train_losses = []
    val_losses = []

    # Training loop with early stopping
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X)

            loss = criterion(outputs, batch_y)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)

        print(f"Epoch [{epoch + 1}/{max_epochs}], Train Loss: {train_loss:.6f}")

        # Step the learning rate scheduler
        scheduler.step()

        # Evaluate on val set
        model.eval()
        with torch.no_grad():
            val_loss = 0.0

            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X)

                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

            # Average across batches
            val_loss = val_loss / len(val_loader)
            val_losses.append(val_loss)  # optional: save per appliance losses
            print('Validation Loss: {:.8f}'.format(val_loss))

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return model, train_losses, val_losses


def evaluate_model(model, test_loader, target_scaler, device):
    model.eval()

    predictions_all = []
    y_test_all = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            batch_predictions = model(batch_X)

            predictions_all.append(batch_predictions.cpu())
            y_test_all.append(batch_y.cpu())

    predictions_tensor = torch.cat(predictions_all, dim=0)
    y_test_tensor = torch.cat(y_test_all, dim=0)

    pred_np = predictions_tensor.reshape(-1).numpy()
    y_test_np = y_test_tensor.reshape(-1).numpy()

    predictions_denorm = target_scaler.inverse_transform(pred_np.reshape(-1, 1)).flatten()
    y_test_denorm = target_scaler.inverse_transform(y_test_np.reshape(-1, 1)).flatten()

    mae = np.mean(np.abs(predictions_denorm - y_test_denorm))
    mse = np.mean((predictions_denorm - y_test_denorm) ** 2)
    rmse = np.sqrt(mse)
    sae = np.abs(np.sum(predictions_denorm) - np.sum(y_test_denorm)) / (np.sum(y_test_denorm) + 1e-8)
    nde = np.linalg.norm(predictions_denorm - y_test_denorm) / (np.linalg.norm(y_test_denorm) + 1e-8)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "SAE": sae,
        "NDE": nde,
    }


def run_training_and_evaluation(hidden_size, sequence_length, stride, num_layers, dataset,
                                     input_columns, device, appliance_name, target_scaler, batch_size=128):
    print(f"Training and evaluating model for: {appliance_name}")

    # Load dataset
    appliance_df = dataset
    train_loader, val_loader, test_loader, seq_len, input_size, output_size = create_dataloader_for_appliance(
        dataset=appliance_df,
        input_columns=input_columns,
        target_appliance=[appliance_name],
        sequence_length=sequence_length,
        stride=stride,
        batch_size=batch_size,
        seed=SEED
    )

    # Train and save model
    model, train_losses, val_losses = train_nilm_model(
        train_loader=train_loader,
        val_loader=val_loader,
        input_size=input_size,
        output_size=output_size,
        device=device,
        hidden_size=hidden_size,
        max_epochs=300,
        patience=10,
        seed=SEED,
        num_layers=num_layers,
    )

    # Evaluate model
    print(f"\nEvaluating model for: {appliance_name}")
    results = evaluate_model(model, test_loader, target_scaler, device)

    return results


def main():
    device = select_device()

    # Load dataset
    dataset = pd.read_parquet('training_data.parquet')

    timestamp_column = dataset.columns[0]
    household_id_column = dataset.columns[1]
    smart_meter_columns = dataset.columns[2:11]
    appliance = 'Other'

    valid_rows = dataset[dataset[appliance] != 0]
    valid_households = valid_rows['household_id'].unique().tolist()

    subset = dataset[dataset['household_id'].isin(valid_households)]
    cols_to_keep = [timestamp_column] + [household_id_column] + list(smart_meter_columns) + [appliance]
    dataset = subset[cols_to_keep].reset_index(drop=True)

    input_columns = list(smart_meter_columns)

    input_scaler = MinMaxScaler()
    dataset[input_columns] = input_scaler.fit_transform(dataset[input_columns])

    target_scaler = MinMaxScaler()
    dataset[appliance] = target_scaler.fit_transform(dataset[[appliance]])

    hidden_sizes = [128, 256, 512]
    sequence_lengths = [120, 360, 720]
    strides = [0.25, 0.5]
    num_layers = [2, 3, 4, 5]

    param_grid = list(
        itertools.product(hidden_sizes, sequence_lengths, strides, num_layers))
    total_runs = len(param_grid)
    results = []

    for i, (hidden_size, seq_length, stride, num_layers) in enumerate(param_grid, 1):
        print(
            f"\nRun {i}/{total_runs}: hidden={hidden_size}, seq_len={seq_length}, stride={stride}, num_layers={num_layers}")

        start = time.time()
        eval_result = run_training_and_evaluation(hidden_size, seq_length, stride, num_layers, dataset, input_columns, device, appliance, target_scaler=target_scaler)

        mae, mse, rmse, sae, nde = eval_result['MAE'], eval_result['MSE'], eval_result['RMSE'], eval_result['SAE'], eval_result['NDE']

        end = time.time()
        print(f"Run {i}/{total_runs} completed in {end - start:.2f} seconds with: {eval_result}")
        results.append(
            {'hidden_size': hidden_size, 'seq_length': seq_length, 'stride': stride, 'num_layers':num_layers, 'eval_result': mae})

    df_results = pd.DataFrame(results)
    df_results.sort_values(by='eval_result', ascending=True, inplace=True)
    print(df_results)


if __name__ == '__main__':
    main()