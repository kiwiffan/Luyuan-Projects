import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Data loading
input_path = './jane-street-real-time-market-data-forecasting'
df_all = pd.read_parquet(f'{input_path}/train.parquet')

df_length = len(df_all)
# df = df_all.iloc[37072420:((df_length // 102410) - 4) * 102410]
df = df_all.iloc[26216960:((df_length // 102410) - 4) * 102410]
df = df.fillna(0)
# Prepare features and normalize
feature_names = [f"feature_{i:02d}" for i in range(79)]
mean_values = df[feature_names + ['responder_6']].mean()
std_values = df[feature_names + ['responder_6']].std()
df.loc[:, feature_names + ['responder_6']] = (df[feature_names + ['responder_6']] - mean_values) / std_values
df.loc[:, feature_names + ['responder_6']] = df[feature_names + ['responder_6']].fillna(0)
mean_df = mean_values.to_frame(name='Mean Values')
std_df = std_values.to_frame(name='Std Values')
mean_df.to_csv('normalization_mean_0106.csv', index=True)
std_df.to_csv('normalization_std_0106.csv', index=True)

# Helper function for creating DataLoader
def chunked_data_loader(df, feature_names, target_name, weight_name, chunk_size=102410, batch_size=1024, timestep=10):
    grouped = df.groupby('symbol_id')
    for name, group in grouped:
        num_chunks = len(group) // chunk_size
        for chunk_index in range(num_chunks + 1):
            start = chunk_index * chunk_size
            end = min(start + chunk_size, len(group))
            chunk = group.iloc[start:end]
            X, y, weights, symbol_ids = [], [], [], []
            for i in range(len(chunk) - timestep):
                X.append(chunk[feature_names].iloc[i:i + timestep].values)
                y.append(chunk[target_name].iloc[i + timestep])
                weights.append(chunk[weight_name].iloc[i + timestep])
                symbol_ids.append(chunk['symbol_id'].iloc[i])

            dataset = TensorDataset(torch.tensor(np.array(X, dtype=np.float32)),
                                    torch.tensor(np.array(y, dtype=np.float32)),
                                    torch.tensor(np.array(weights, dtype=np.float32)),
                                    torch.tensor(np.array(symbol_ids, dtype=np.int64)))
            yield DataLoader(dataset, batch_size=batch_size, shuffle=False)

# GRU Model Definition
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, embedding_dim, num_symbols):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(num_symbols, embedding_dim)
        self.gru = nn.GRU(input_dim + embedding_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, symbol_ids):
        symbol_embeds = self.embedding(symbol_ids)
        symbol_embeds = symbol_embeds.unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat((x, symbol_embeds), dim=-1)
        h, _ = self.gru(x)
        h_last = h[:, -1, :]
        h = self.relu(self.fc1(h_last))
        output = self.fc2(h)
        return output

# Custom Loss Function: Weighted R^2 Loss
class WeightedR2Loss(nn.Module):
    def __init__(self):
        super(WeightedR2Loss, self).__init__()

    def forward(self, y_true, y_pred, weights):
        wrss = torch.sum(weights * (y_true - y_pred) ** 2)
        wtss = torch.sum(weights * y_true ** 2)
        return wrss / (wtss + 1e-8)

# Helper function to create validation sequences
def create_valid_sequences(df, feature_names, target_name, weight_name, timestep):
    grouped = df.groupby('symbol_id')
    X_valid = []
    y_valid = []
    w_valid = []
    symbol_ids_valid = []  # Collecting symbol_ids for validation

    for _, group in grouped:
        for i in range(len(group) - timestep):
            X_valid.append(group[feature_names].iloc[i:i + timestep].values)
            y_valid.append(group[target_name].iloc[i + timestep])
            w_valid.append(group[weight_name].iloc[i + timestep])
            symbol_ids_valid.append(group['symbol_id'].iloc[i])  # Collect corresponding symbol_id

    return X_valid, y_valid, w_valid, symbol_ids_valid  # Return data including symbol_ids

# Validation data preparation
print('Valid process started')
valid_data = df_all.iloc[((df_length // 102410) - 4) * 102410:]
valid_data.loc[:, feature_names + ['responder_6']] = (valid_data[feature_names + ['responder_6']] - mean_values) / std_values
valid_data.loc[:, feature_names + ['responder_6']] = valid_data[feature_names + ['responder_6']].fillna(0)
X_valid, y_valid, w_valid, symbol_ids_valid = create_valid_sequences(valid_data, feature_names, 'responder_6', 'weight', timestep=10)
X_valid = torch.tensor(np.array(X_valid, dtype=np.float32))
y_valid = torch.tensor(np.array(y_valid, dtype=np.float32))
w_valid = torch.tensor(np.array(w_valid, dtype=np.float32))
symbol_ids_valid = torch.tensor(np.array(symbol_ids_valid, dtype=np.int64))  # Convert symbol_ids_valid to tensor
print('Valid process finished')
# Training and validation in training function
def train_with_chunks():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRUModel(input_dim=79, hidden_dim=256, output_dim=1, embedding_dim=10, num_symbols=df['symbol_id'].max() + 1).to(device)
    criterion = WeightedR2Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_losses = []
    val_losses = []

    for epoch in range(1):  # Adjust number of epochs as needed
        for i, chunk_loader in tqdm(enumerate(chunked_data_loader(df, feature_names, 'responder_6', 'weight'))):
            model.train()
            chunk_loss = 0.0
            for X_batch, y_batch, w_batch, symbol_ids_batch in tqdm(chunk_loader):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                w_batch = w_batch.to(device)
                symbol_ids_batch = symbol_ids_batch.to(device)

                optimizer.zero_grad()
                y_pred = model(X_batch, symbol_ids_batch).squeeze()
                loss = criterion(y_batch, y_pred, w_batch)
                loss.backward()
                optimizer.step()
                chunk_loss += loss.item()
            train_losses.append(chunk_loss / len(chunk_loader))

            # Validation
            model.eval()
            with torch.no_grad():
                y_valid_pred = model(X_valid.to(device), symbol_ids_valid.to(device)).squeeze()
                val_loss = criterion(y_valid.to(device), y_valid_pred, w_valid.to(device)).item()
            val_losses.append(val_loss)
            print(f'Chunk [{i+1}], Train Loss: {chunk_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Plot training and validation losses
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Chunks')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()
    plt.savefig('training_validation_loss_0106.png')
    plt.close()

    # Save the model
    torch.save(model.state_dict(), 'gru0106.pth')

train_with_chunks()
