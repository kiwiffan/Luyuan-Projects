import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Data loading
input_path = './jane-street-real-time-market-data-forecasting'
df_all = pd.read_parquet(f'{input_path}/train.parquet')
# lags_df = pd.read_parquet('jane-street-real-time-market-data-forecasting/lags.parquet/date_id=0/part-0.parquet')
feature_names = [f"feature_{i:02d}" for i in range(79)]
# df_all = reduce_mem_usage(df_all)
# lags_df = reduce_mem_usage(lags_df)

lag_columns = ['date_id', 'symbol_id', 'responder_0', 'responder_1', 'responder_2', 'responder_3', 'responder_4', 'responder_5', 'responder_6', 'responder_7', 'responder_8']
lag_dataset = df_all[lag_columns]
lag_data = lag_dataset.groupby(['symbol_id', 'date_id']).mean().groupby(level=0).shift()
lag_data = lag_data.reset_index()
lag_data.columns = ['symbol_id', 'date_id'] + [f'responder_{i}_lag_1' for i in range(9)]
df_all = pd.merge(df_all, lag_data, on=['symbol_id', 'date_id'], how='left')
# mask_total_zero = (df['time_id'] == 0) & (df['date_id'] == 0)
mask_time_zero = (df_all['time_id'] == 0)
for col in [col for col in df_all.columns if 'lag_' in col]:
    df_all[col] = np.where(mask_time_zero, df_all[col], 0)

# train_dates = df_all['date_id'].unique()[:-10]
df_length = len(df_all)
df = df_all.iloc[37076040:((df_length // 102420) - 4) * 102420]

feature_columns = feature_names + [f'responder_{k}_lag_1' for k in range(9)]
mean_values = df[feature_columns + ['responder_6']].mean()
std_values = df[feature_columns + ['responder_6']].std()
df.loc[:, feature_columns + ['responder_6']] = (df[feature_columns + ['responder_6']] - mean_values) / std_values
df.loc[:, feature_columns + ['responder_6']] = df[feature_columns + ['responder_6']].fillna(0)
mean_df = mean_values.to_frame(name='Mean Values')
std_df = std_values.to_frame(name='Std Values')
mean_df.to_csv('trans_normalization_mean.csv', index=True)
std_df.to_csv('trans_normalization_std.csv', index=True)

def chunked_data_loader(df, feature_names, target_name, weight_name, chunk_size=102420, batch_size=512, timestep=20):
    grouped = df.groupby('symbol_id')

    for name, group in grouped:
        num_chunks = len(group) // chunk_size
        for chunk_index in range(num_chunks + 1):
            start = chunk_index * chunk_size
            end = min(start + chunk_size, len(group))
            chunk = group.iloc[start:end]

            X = []
            y = []
            weights = []

            for i in range(len(chunk) - timestep):
                X.append(chunk[feature_names + ['symbol_id'] + [f'responder_{k}_lag_1' for k in range(9)]].iloc[i:i + timestep].values)
                y.append(chunk[target_name].iloc[i + timestep])
                weights.append(chunk[weight_name].iloc[i + timestep])

            # Convert to tensors
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            weights = np.array(weights, dtype=np.float32)
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            weights = torch.tensor(weights, dtype=torch.float32)
            dataset = TensorDataset(X, y, weights)
            yield DataLoader(dataset, batch_size=batch_size, shuffle=False)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        encoding = self.encoding[:, :x.size(1)].to(x.device)
        return x + encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        encoding = self.encoding[:, :x.size(1)].to(x.device)
        return x + encoding

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        output = self.fc_out(x[:, -1, :])
        return output


class WeightedR2Loss(nn.Module):
    def __init__(self):
        super(WeightedR2Loss, self).__init__()

    def forward(self, y_true, y_pred, weights):
        wrss = torch.sum(weights * (y_true - y_pred) ** 2)
        wtss = torch.sum(weights * y_true ** 2)
        return wrss / (wtss + 1e-8)

# Initialize and prepare the validation data
print('Valid process started')

def create_valid_sequences(df, feature_names, target_name, weight_name, timestep):
    grouped = df.groupby('symbol_id')
    X_valid = []
    y_valid = []
    w_valid = []

    for _, group in grouped:
        for i in range(len(group) - timestep):
            X_valid.append(group[feature_names + ['symbol_id'] + [f'responder_{k}_lag_1' for k in range(9)]].iloc[
                           i:i + timestep].values)
            y_valid.append(group[target_name].iloc[i + timestep])
            w_valid.append(group[weight_name].iloc[i + timestep])

    return X_valid, y_valid, w_valid

valid_data = df_all.iloc[((df_length // 102420) - 4) * 102420:]
valid_data.loc[:, feature_columns + ['responder_6']] = (valid_data[feature_columns + ['responder_6']] - mean_values) / std_values
valid_data.loc[:, feature_columns + ['responder_6']] = valid_data[feature_columns + ['responder_6']].fillna(0)
X_valid, y_valid, w_valid = create_valid_sequences(valid_data, feature_names, 'responder_6', 'weight', timestep=10)
X_valid = torch.tensor(np.array(X_valid, dtype=np.float32), dtype=torch.float32)
y_valid = torch.tensor(np.array(y_valid, dtype=np.float32), dtype=torch.float32)
w_valid = torch.tensor(np.array(w_valid, dtype=np.float32), dtype=torch.float32)

print('Valid process finished')

def train_with_chunks():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(input_dim=89, d_model=256, nhead=8, num_encoder_layers=4, output_dim=1).to(device)
    criterion = WeightedR2Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    train_losses = []
    val_losses = []

    for epoch in range(1):
        for i, chunk_loader in tqdm(enumerate(chunked_data_loader(df, feature_names, 'responder_6', 'weight'))):
            model.train()
            chunk_loss = 0.0
            for X_batch, y_batch, w_batch in tqdm(chunk_loader):
                X_batch, y_batch, w_batch = (
                    X_batch.to(device),
                    y_batch.to(device),
                    w_batch.to(device),
                )
                optimizer.zero_grad()
                y_pred = model(X_batch).squeeze()
                loss = criterion(y_batch, y_pred, w_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                chunk_loss += loss.item()
            train_losses.append(chunk_loss / len(chunk_loader))

            model.eval()
            with torch.no_grad():
                y_valid_pred = model(X_valid.to(device)).squeeze()
                val_loss = criterion(y_valid.to(device), y_valid_pred, w_valid.to(device)).item()
            val_losses.append(val_loss)
            print(f'Chunk [{i+1}], Train Loss: {chunk_loss:.4f}, Validation Loss: {val_loss:.4f}')

    torch.save(model.state_dict(), 'transformer1223.pth')

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Chunks')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()
    plt.savefig('trans1223.png')
    plt.close()

train_with_chunks()

