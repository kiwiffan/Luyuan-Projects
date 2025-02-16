import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# def reduce_mem_usage(df, float16_as32=True):
#     start_mem = df.memory_usage().sum() / 1024 ** 2
#     print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
#
#     for col in df.columns:
#         col_type = df[col].dtype
#         if col_type != object and str(col_type) != 'category':
#             c_min, c_max = df[col].min(), df[col].max()
#             if str(col_type)[:3] == 'int':
#                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
#                     df[col] = df[col].astype(np.int8)
#                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
#                     df[col] = df[col].astype(np.int16)
#                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
#                     df[col] = df[col].astype(np.int32)
#                 elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
#                     df[col] = df[col].astype(np.int64)
#             else:
#                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
#                     if float16_as32:
#                         df[col] = df[col].astype(np.float32)
#                     else:
#                         df[col] = df[col].astype(np.float16)
#                 elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
#                     df[col] = df[col].astype(np.float32)
#                 else:
#                     df[col] = df[col].astype(np.float64)
#     end_mem = df.memory_usage().sum() / 1024 ** 2
#     print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
#     print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
#     return df

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
df = df_all.iloc[37072420:((df_length // 102410) - 4) * 102410]

feature_columns = feature_names + [f'responder_{k}_lag_1' for k in range(9)]
mean_values = df[feature_columns + ['responder_6']].mean()
std_values = df[feature_columns + ['responder_6']].std()
df.loc[:, feature_columns + ['responder_6']] = (df[feature_columns + ['responder_6']] - mean_values) / std_values
df.loc[:, feature_columns + ['responder_6']] = df[feature_columns + ['responder_6']].fillna(0)
mean_df = mean_values.to_frame(name='Mean Values')
std_df = std_values.to_frame(name='Std Values')
mean_df.to_csv('normalization_mean.csv', index=True)
std_df.to_csv('normalization_std.csv', index=True)

# Data chunking function
def chunked_data_loader(df, feature_names, target_name, weight_name, chunk_size=102410, batch_size=1024, timestep=10):
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


# PyTorch Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 通过LSTM层
        h, _ = self.lstm(x)
        # 获取LSTM的最后一个时间步的输出
        h_last = h[:, -1, :]
        # 通过第一个全连接层
        h = self.relu(self.fc1(h_last))
        # 通过第二个全连接层获得最终输出
        output = self.fc2(h)
        return output

class WeightedR2Loss(nn.Module):
    def __init__(self):
        super(WeightedR2Loss, self).__init__()

    def forward(self, y_true, y_pred, weights):
        wrss = torch.sum(weights * (y_true - y_pred) ** 2)
        wtss = torch.sum(weights * y_true ** 2)
        return wrss / (wtss + 1e-8)

# Validation data
# def prepare_validation_data(df, feature_names, timestep, valid_dates):
#     valid_data = df.loc[df['date_id'].isin(valid_dates)]
#     X_valid = []
#
#     for i in range(len(valid_data) - timestep + 1):
#         X_slice = valid_data.iloc[i:i + timestep]
#         features = X_slice[feature_names + ['symbol_id'] + [f'responder_{k}_lag_1' for k in range(9)]]
#         features_filled = features.fillna(0).values
#         X_valid.append(features_filled)
#
#     return X_valid
#
# valid_dates = df_all['date_id'].unique()[-10:]
# X_valid = prepare_validation_data(df_all, feature_names, timestep=10, valid_dates=valid_dates)

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


valid_data = df_all.iloc[((df_length // 102410) - 4) * 102410:]
valid_data.loc[:, feature_columns + ['responder_6']] = (valid_data[feature_columns + ['responder_6']] - mean_values) / std_values
valid_data.loc[:, feature_columns + ['responder_6']] = valid_data[feature_columns + ['responder_6']].fillna(0)
X_valid, y_valid, w_valid = create_valid_sequences(valid_data, feature_names, 'responder_6', 'weight', timestep=10)
X_valid = torch.tensor(np.array(X_valid, dtype=np.float32), dtype=torch.float32)
y_valid = torch.tensor(np.array(y_valid, dtype=np.float32), dtype=torch.float32)
w_valid = torch.tensor(np.array(w_valid, dtype=np.float32), dtype=torch.float32)

print('Valid process finished')

# Training function
def train_with_chunks():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_dim=89, hidden_dim=256, output_dim=1).to(device)
    criterion = WeightedR2Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    train_losses = []
    val_losses = []

    for epoch in range(1):
        # epoch_loss = 0.0

        # Iterate over chunks
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
        # train_losses.append(epoch_loss)
        # Validation
            model.eval()
            with torch.no_grad():
                y_valid_pred = model(X_valid.to(device)).squeeze()
                val_loss = criterion(y_valid.to(device), y_valid_pred, w_valid.to(device)).item()
            val_losses.append(val_loss)
            print(f'Chunk [{i+1}], Train Loss: {chunk_loss:.4f}, Validation Loss: {val_loss:.4f}')

    torch.save(model.state_dict(), 'lstm1224.pth')

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Chunks')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()
    plt.savefig('lstm1224.png')
    plt.close()

train_with_chunks()
