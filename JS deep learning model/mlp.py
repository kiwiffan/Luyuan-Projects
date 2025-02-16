import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from typing import Tuple, List, Iterator
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import gc
import math
import time
from pathlib import Path
import json
import os


# torch.set_default_dtype(torch.float64)

def weighted_r2_score(y_true, y_pred, weights):
    numerator = np.sum(weights * (y_true - y_pred) ** 2)
    denominator = np.sum(weights * y_true ** 2)
    r2 = 1 - (numerator / denominator + 1e-20)
    return r2


@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 256
    chunks_size: int = 100000
    epochs: int = 1000
    patience: int = 10
    hidden_layers: List[int] = None
    log_dir: str = "runs/experiment"

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [100, 50]
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = f"{self.log_dir}_{timestamp}"


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        loss = torch.mean(weights * (outputs - targets) ** 2) / (torch.mean(weights * targets ** 2) + 1e-16)
        return loss


class ChunkDataLoader:
    def __init__(self, file_path: str, chunk_size: int):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.total_rows = None
        self.n_features = None
        self.feature_columns = None
        self._get_total_rows_and_features()

    def _get_total_rows_and_features(self):
        df = pl.scan_parquet(self.file_path)
        schema = df.collect_schema()
        all_columns = list(schema.keys())

        self.feature_columns = list(all_columns[4:-10])
        self.n_features = len(self.feature_columns)

        self.total_rows = df.select(pl.len()).collect().item()

        print(f"Total number of rows: {self.total_rows:,}")
        print(f"Number of features: {self.n_features}")

    def get_chunks(self) -> Iterator[pl.DataFrame]:
        num_chunks = math.ceil(self.total_rows / self.chunk_size)
        for i in range(num_chunks):
            start_row = i * self.chunk_size
            df_chunk = (pl.scan_parquet(self.file_path)
                        .slice(start_row, self.chunk_size)
                        .collect())
            yield df_chunk

    def get_feature_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        sum_stats = np.zeros((2, self.n_features), dtype=np.float64)  # [sum, sum_sq]
        n_samples = 0

        print("\nCalculating feature statistics:")
        for chunk in tqdm(self.get_chunks(), desc="Processing chunks"):
            X = chunk.select(self.feature_columns).to_numpy()
            X = X.astype(np.float32)
            sum_stats[0] += np.sum(X, axis=0)
            sum_stats[1] += np.sum(X * X, axis=0)
            n_samples += len(X)

            if n_samples % (self.chunk_size * 10) == 0:
                print(f"Processed {n_samples:,} samples")

        mean = sum_stats[0] / n_samples
        std = np.sqrt(sum_stats[1] / n_samples - (mean * mean))
        std = np.maximum(std, 1e-6)

        print("\nFeature statistics calculation completed.")
        return mean, std


class CustomDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
        self.X = X
        self.y = y
        self.w = w

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_layers: List[int]):
        super(MLP, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                # nn.BatchNorm1d(hidden_size),
                # nn.Dropout(0.2)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ProgressTracker:
    def __init__(self, config: TrainingConfig):
        self.writer = SummaryWriter(config.log_dir)
        self.global_step = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'val_r2': []
        }

    def update_train_metrics(self, loss: float, lr: float, epoch: int, chunk_idx: int):
        self.writer.add_scalar('Training/Loss', loss, self.global_step)
        self.writer.add_scalar('Training/Learning_Rate', lr, self.global_step)
        self.writer.add_scalar('Training/Epoch', epoch, self.global_step)
        self.writer.add_scalar('Training/Chunk', chunk_idx, self.global_step)
        self.global_step += 1

    def update_validation_metrics(self, val_loss: float, rmse: float, r2: float, epoch: int):
        self.writer.add_scalar('Validation/Loss', val_loss, epoch)
        self.writer.add_scalar('Validation/RMSE', rmse, epoch)
        self.writer.add_scalar('Validation/R2', r2, epoch)

        self.history['val_loss'].append(val_loss)
        self.history['val_rmse'].append(float(rmse))
        self.history['val_r2'].append(float(r2))

    def save_history(self):
        history_path = Path(self.writer.log_dir) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)

    def close(self):
        self.save_history()
        self.writer.close()


class MLPTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.tracker = ProgressTracker(config)
        self.feature_mean = None
        self.feature_std = None
        self.feature_columns = None
        self.n_features = None
        self.current_lr = None

    def _log_lr_change(self, old_lr: float, new_lr: float):
        if old_lr != new_lr:
            print(f"\nLearning rate changed: {old_lr:.6f} -> {new_lr:.6f}")

    def prepare_validation_data(self, data_chunk: pl.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        X = data_chunk.select(data_chunk.columns[4:-10]).to_numpy()
        y = data_chunk.select(['responder_6']).to_numpy().ravel()

        return (torch.tensor(X, dtype=torch.float32).to(self.device),
                torch.tensor(y, dtype=torch.float32).to(self.device))

    def train_on_chunk(self, data_chunk: pl.DataFrame,
                       model: nn.Module,
                       optimizer: optim.Optimizer,
                       criterion: nn.Module,
                       epoch: int,
                       chunk_idx: int) -> float:

        X = data_chunk.select(data_chunk.columns[4:-10]).to_numpy()
        y = data_chunk.select(['responder_6']).to_numpy().reshape(-1, 1)
        w = data_chunk.select(['weight']).to_numpy().reshape(-1, 1)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        w_tensor = torch.tensor(w, dtype=torch.float32)

        dataset = CustomDataset(X_tensor, y_tensor, w_tensor)

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(dataloader,
                            desc=f'Epoch {epoch + 1}/{self.config.epochs}, Chunk {chunk_idx}',
                            leave=False)

        model.train()
        for batch_X, batch_y, batch_w in progress_bar:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_w = batch_w.to(self.device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y, batch_w)
            loss.backward()
            optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']
            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.6f}'
            })

            self.tracker.update_train_metrics(
                loss.item(),
                current_lr,
                epoch,
                chunk_idx
            )

        del dataset, dataloader, X_tensor, y_tensor, X, y
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return total_loss / num_batches

    def prepare_validation_data(self, data_chunk: pl.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        X = data_chunk.select(data_chunk.columns[4:-10]).to_numpy()
        y = data_chunk.select(['responder_6']).to_numpy().ravel()

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        return X_tensor, y_tensor

    def train(self, data_path: str):
        chunk_loader = ChunkDataLoader(data_path, self.config.chunks_size)
        self.feature_columns = chunk_loader.feature_columns
        self.n_features = chunk_loader.n_features
        print("Computing feature statistics...")
        self.feature_mean, self.feature_std = chunk_loader.get_feature_stats()

        first_chunk = next(chunk_loader.get_chunks())
        input_size = len(first_chunk.columns[4:-10])

        model = MLP(input_size, self.config.hidden_layers).to(self.device)
        criterion = WeightedMSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
        )

        model_path = 'model_checkpoint.pth'
        if os.path.exists(model_path):
            print("Loading saved model...")
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 开始
            best_loss = checkpoint['loss']
            print(f"Model loaded, resuming training from epoch {start_epoch}")
        else:
            start_epoch = 0  # 从头开始训练
            best_loss = float('inf')

        self.current_lr = self.config.learning_rate

        val_chunk = first_chunk.sample(n=min(10000, len(first_chunk)))

        best_loss = float('inf')
        epochs_no_improve = 0

        epoch_progress = tqdm(range(start_epoch, self.config.epochs), desc='Training Progress')
        for epoch in epoch_progress:
            epoch_loss = 0
            num_chunks = 0

            for chunk_idx, chunk in enumerate(chunk_loader.get_chunks()):
                chunk_loss = self.train_on_chunk(
                    chunk, model, optimizer, criterion, epoch, chunk_idx
                )
                epoch_loss += chunk_loss
                num_chunks += 1

            avg_epoch_loss = epoch_loss / num_chunks

            model.eval()
            X_val = val_chunk.select(self.feature_columns).to_numpy()
            y_val = val_chunk.select(['responder_6']).to_numpy().reshape(-1, 1)
            w_val = val_chunk.select(['weight']).to_numpy().reshape(-1, 1)

            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)
            w_val_tensor = torch.tensor(w_val, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor, w_val_tensor).item()

                val_pred = val_outputs.cpu().numpy()
                val_rmse = root_mean_squared_error(y_val, val_pred)
                val_r2 = weighted_r2_score(y_val, val_pred, w_val)

            self.tracker.update_validation_metrics(
                val_loss, val_rmse, val_r2, epoch
            )

            scheduler.step(val_loss)

            epoch_progress.set_postfix({
                'train_loss': f'{avg_epoch_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'val_rmse': f'{val_rmse:.4f}',
                'val_r2': f'{val_r2:.4f}'
            })

            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                model_save_path = Path(self.config.log_dir) / 'best_model.pth'
                model_save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'feature_mean': self.feature_mean,
                    'feature_std': self.feature_std,
                }, str(model_save_path))
            else:
                epochs_no_improve += 1

            if epochs_no_improve == self.config.patience:
                print(f'\nEarly stopping at epoch {epoch}')
                break

        self.tracker.close()
        print(f"Training completed. Model saved at: {self.config.log_dir}")


def main():
    config = TrainingConfig(
        learning_rate=0.000001,
        batch_size=256,
        chunks_size=1000000,
        epochs=1000,
        patience=20,
        hidden_layers=[2 ** (10 - i) for i in range(10)],
        log_dir="runs/mlp_experiment"
    )

    trainer = MLPTrainer(config)
    trainer.train('train_scaled_no_outliers.parquet')


if __name__ == "__main__":
    main()
