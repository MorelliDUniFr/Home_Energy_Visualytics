# Data manipulation
import pandas as pd
import itertools

# PyTorch core
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR

# Preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SEED = 42

class SlidingWindowDataset(Dataset):
    def __init__(self, data, input_columns, target_columns, sequence_length, stride=1, time_threshold='10s'):
        self.sequence_length = sequence_length
        self.stride = stride
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
            for start in range(0, len(group) - sequence_length + 1, stride):
                self.indices.append((group, start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        group, start = self.indices[idx]
        x = group[self.input_columns].iloc[start:start + self.sequence_length].values
        y = group[self.target_columns].iloc[start + self.sequence_length - 1].values

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y


class NILMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, bidirectional=False,
                 use_attention=False, dropout_rate=0.3, use_batch_norm=False):
        super(NILMModel, self).__init__()

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # LSTM layer (can be bidirectional, with layer normalization)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=0.2)

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

        # Optional batch normalization for LSTM output
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.batch_norm_lstm = nn.BatchNorm1d(hidden_size * (2 if bidirectional else 1))

        # Residual connection (optional)
        self.residual_fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Optional activation function (ReLU)
        self.relu = nn.ReLU()

        # Attention mechanism (Optional for NILM)
        if self.use_attention:
            self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, batch_first=True)

    def forward(self, x):
        device = x.device  # Get the device of the input tensor
        # LSTM output
        lstm_out, (hn, cn) = self.lstm(x)

        # Ensure hidden state is on the same device as input
        hn = hn.to(device)
        cn = cn.to(device)

        # Apply batch normalization to LSTM output if specified
        if self.use_batch_norm:
            lstm_out = self.batch_norm_lstm(lstm_out[:, -1, :])  # Apply normalization on last step only

        # Attention mechanism to focus on important time steps
        if self.use_attention:
            attention_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        else:
            attention_out = lstm_out

        # Residual connection
        residual_out = self.residual_fc(attention_out[:, -1, :])

        # Dropout for regularization
        out = self.dropout(residual_out)

        # Optional activation function (ReLU)
        out = self.relu(out)

        return out


class SoftF1Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(SoftF1Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, labels):
        # Sigmoid for binary classification
        probs = torch.sigmoid(logits)
        labels = labels.float()

        # Calculate TP, FP, FN
        tp = (probs * labels).sum(dim=0)
        fp = (probs * (1 - labels)).sum(dim=0)
        fn = ((1 - probs) * labels).sum(dim=0)

        soft_f1 = 2 * tp / (2 * tp + fp + fn + self.epsilon)
        loss = 1 - soft_f1.mean()
        return loss


def select_device():
    # Check for GPU availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def train_model(model, train_loader, val_loader, device, loss_fn, step_size, gamma):
    # Loss function and optimizer
    criterion = loss_fn
    optimizer = Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Initialize variables for tracking
    num_epochs = 80
    best_loss = float('inf')
    patience = 5  # Early stopping patience
    patience_counter = 0

    # To track loss
    train_losses = []
    val_losses = []

    # Training loop with early stopping
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X)

            # Compute loss
            loss = criterion(outputs, batch_y)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)

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

            val_loss = val_loss / len(val_loader)
            val_losses.append(val_loss)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}/80")
                break


def model_evaluation(model, test_loader, device):
    # Evaluation phase with batching
    model.eval()
    predictions_all = []
    y_test_all = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            preds = model(batch_X)
            predictions_all.append(preds.cpu())
            y_test_all.append(batch_y.cpu())

    # Concatenate and compute MAE
    predictions_tensor = torch.cat(predictions_all)
    y_test_tensor = torch.cat(y_test_all)

    # Compute MSE
    mse = nn.MSELoss()(predictions_tensor, y_test_tensor).item()
    return mse


def run_experiment(loss_fn, hidden_size, step_size, gamma, dropout_rate, sequence_length, stride, use_attention, num_layers, dataset, input_columns, target_appliances, device):
    dataset = SlidingWindowDataset(dataset, input_columns, target_appliances, sequence_length=sequence_length, stride=stride)

    # Train/val/test split on the full dataset with sequences
    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len

    # Split the dataset randomly
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_len, val_len, test_len], generator=generator)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=True)  # shuffle=True for training
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Get one sample to determine input and output dimensions
    sample_X, sample_y = train_dataset[0]
    input_size = sample_X.shape[1]  # Number of input features
    output_size = sample_y.shape[0]  # Number of target appliances

    model = NILMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, dropout_rate=dropout_rate, use_attention=use_attention).to(device)

    # Train the model
    train_model(model, train_loader, val_loader, device=device, loss_fn=loss_fn, step_size=step_size, gamma=gamma)

    # Evaluate the model
    eval_results = model_evaluation(model, test_loader, device)

    return eval_results

def main():
    device = select_device()

    # Load dataset
    dataset = pd.read_parquet('training_data.parquet')

    smart_meter_columns = dataset.columns[2:11]
    appliances_columns = dataset.columns[11:]

    # Sliding window dataset creation
    input_columns = list(smart_meter_columns)
    target_appliances = appliances_columns

    # Normalize the dataset
    # input_scaler = StandardScaler()
    # input_scaler = load('input_scaler.pkl')  # Load pre-trained scaler if available
    input_scaler = StandardScaler()
    dataset[input_columns] = input_scaler.fit_transform(dataset[input_columns])

    # Normalize target appliances with one scaler per appliance
    target_scalers = {}
    for appliance in target_appliances:
        scaler = MinMaxScaler()
        dataset[appliance] = scaler.fit_transform(dataset[[appliance]])
        target_scalers[appliance] = scaler
    # targets_scaler = load('target_scalers.pkl')  # Load pre-trained scaler if available

    # target_scaler = MinMaxScaler()
    # dataset[target_appliances] = target_scaler.fit_transform(dataset[target_appliances])

    # Start loop for grid search
    # losses_list = [nn.L1Loss(), nn.MSELoss(), nn.SmoothL1Loss()]
    losses_list = [nn.MSELoss()]
    hidden_sizes = [64, 128, 256]
    step_sizes = [15]
    gammas = [0.9]
    dropouts = [0.3]
    sequence_lengths = [60, 120, 240]
    strides = [15, 30, 50]
    use_attention = [False]
    num_layers = [2, 3, 4, 5]

    param_grid = list(
        itertools.product(losses_list, hidden_sizes, step_sizes, gammas, dropouts, sequence_lengths, strides, use_attention, num_layers))
    total_runs = len(param_grid)
    results = []

    for i, (loss_fn, hidden_size, step_size, gamma, dropout, seq_length, stride, use_attention, num_layers) in enumerate(param_grid, 1):
        print(
            f"\nRun {i}/{total_runs}: loss={loss_fn.__class__.__name__}, hidden={hidden_size}, step={step_size}, gamma={gamma}, dropout={dropout}, seq_len={seq_length}, stride={stride}, use_attention={use_attention}, num_layers={num_layers}")
        eval_result = run_experiment(loss_fn, hidden_size, step_size, gamma, dropout, seq_length, stride, use_attention, num_layers, dataset,
                                     input_columns, target_appliances, device)
        results.append(
            {'loss_fn': loss_fn.__class__.__name__, 'hidden_size': hidden_size, 'step_size': step_size, 'gamma': gamma,
             'dropout': dropout, 'seq_length': seq_length, 'stride': stride, 'use_attention':use_attention, 'num_layers':num_layers, 'eval_result': eval_result})

    df_results = pd.DataFrame(results)
    df_results.sort_values(by='eval_result', ascending=True, inplace=True)
    print(df_results)


if __name__ == '__main__':
    main()