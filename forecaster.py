"""
Time Series Forecasting with LSTM
A PyTorch implementation for financial time series prediction
Author: Konstantinos Kantoutsis
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class LSTMForecaster(nn.Module):
    """LSTM-based model for time series forecasting."""
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        super(LSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        # Take the last time step
        last_output = lstm_out[:, -1, :]
        # Fully connected layers
        predictions = self.fc(last_output)
        return predictions


class TimeSeriesDataset:
    """Handles data preprocessing and sequence creation."""
    
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def create_sequences(
        self, 
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and targets for training."""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            y.append(data[i])
            
        return np.array(X), np.array(y)
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        train_ratio: float = 0.8
    ) -> Tuple[DataLoader, DataLoader, np.ndarray]:
        """Prepare train and test DataLoaders."""
        
        # Extract and scale data
        data = df[target_col].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        # Train/test split
        train_size = int(len(X) * train_ratio)
        
        X_train = torch.FloatTensor(X[:train_size])
        y_train = torch.FloatTensor(y[:train_size])
        X_test = torch.FloatTensor(X[train_size:])
        y_test = torch.FloatTensor(y[train_size:])
        
        # Create DataLoaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        return train_loader, test_loader, y_test.numpy()


class Trainer:
    """Handles model training and evaluation."""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, np.ndarray]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                
        return total_loss / len(test_loader), np.array(all_predictions)
    
    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 100,
        verbose: bool = True
    ) -> None:
        """Full training loop."""
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, _ = self.evaluate(test_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {train_loss:.6f} "
                      f"Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic financial time series data."""
    np.random.seed(42)
    
    # Create trend + seasonality + noise
    t = np.arange(n_samples)
    trend = 0.001 * t
    seasonality = 0.5 * np.sin(2 * np.pi * t / 50)
    noise = 0.1 * np.random.randn(n_samples)
    
    # Random walk component (like stock prices)
    random_walk = np.cumsum(0.02 * np.random.randn(n_samples))
    
    price = 100 + trend + seasonality + noise + random_walk
    
    return pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', periods=n_samples, freq='D'),
        'close': price
    })


def plot_results(
    actual: np.ndarray,
    predicted: np.ndarray,
    train_losses: List[float],
    val_losses: List[float]
) -> None:
    """Plot training results and predictions."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Predictions vs Actual
    axes[0].plot(actual, label='Actual', alpha=0.7)
    axes[0].plot(predicted, label='Predicted', alpha=0.7)
    axes[0].set_title('Predictions vs Actual Values')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Price (scaled)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Training curves
    axes[1].plot(train_losses, label='Train Loss')
    axes[1].plot(val_losses, label='Validation Loss')
    axes[1].set_title('Training and Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('forecasting_results.png', dpi=150)
    plt.show()


def main():
    """Main execution function."""
    print("=" * 60)
    print("Time Series Forecasting with LSTM")
    print("=" * 60)
    
    # Generate or load data
    print("\n[1] Loading data...")
    df = generate_sample_data(1000)
    print(f"    Data shape: {df.shape}")
    
    # Prepare dataset
    print("\n[2] Preparing sequences...")
    dataset = TimeSeriesDataset(sequence_length=60)
    train_loader, test_loader, y_test = dataset.prepare_data(
        df, target_col='close', train_ratio=0.8
    )
    print(f"    Sequence length: 60")
    print(f"    Train batches: {len(train_loader)}")
    print(f"    Test batches: {len(test_loader)}")
    
    # Initialize model
    print("\n[3] Initializing model...")
    model = LSTMForecaster(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    )
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\n[4] Training...")
    trainer = Trainer(model, learning_rate=0.001)
    trainer.fit(train_loader, test_loader, epochs=100, verbose=True)
    
    # Evaluate
    print("\n[5] Evaluating...")
    val_loss, predictions = trainer.evaluate(test_loader)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"    RMSE: {rmse:.6f}")
    print(f"    MAE:  {mae:.6f}")
    
    # Plot
    print("\n[6] Generating plots...")
    plot_results(y_test, predictions, trainer.train_losses, trainer.val_losses)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
