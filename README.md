# Time Series Forecasting with LSTM

A PyTorch implementation of LSTM-based time series forecasting for financial data.

## Features

- **LSTM Architecture**: Multi-layer LSTM with dropout regularization
- **Modular Design**: Separate classes for model, data handling, and training
- **Early Stopping**: Prevents overfitting with patience-based early stopping
- **Visualization**: Training curves and prediction plots

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from forecaster import LSTMForecaster, TimeSeriesDataset, Trainer

# Prepare data
dataset = TimeSeriesDataset(sequence_length=60)
train_loader, test_loader, y_test = dataset.prepare_data(df, 'close')

# Initialize and train
model = LSTMForecaster(hidden_size=64, num_layers=2)
trainer = Trainer(model, learning_rate=0.001)
trainer.fit(train_loader, test_loader, epochs=100)

# Evaluate
loss, predictions = trainer.evaluate(test_loader)
```

## Quick Start

```bash
python forecaster.py
```

## Model Architecture

```
Input → LSTM (64 units, 2 layers) → FC (32) → ReLU → Dropout → FC (1) → Output
```

## Results

On synthetic financial data:
- **RMSE**: ~0.02
- **MAE**: ~0.015

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas, Scikit-learn, Matplotlib

## Author

Konstantinos Kantoutsis - [GitHub](https://github.com/kkanteea)

## License

MIT License
