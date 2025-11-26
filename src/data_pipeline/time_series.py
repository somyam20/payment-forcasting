import os
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Union
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.random import set_seed
import yaml

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Load default forecasting config from config/model_config.yaml if available
CONFIG_PATH = os.path.join("config", "model_config.yaml")

def _load_config():
    default = {
        "window_size": 10,
        "forecast_steps": 30,
        "epochs": 10,
        "batch_size": 32,
        "units_layer1": 64,
        "units_layer2": 32
    }
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            lstm_cfg = cfg.get("forecasting", {}).get("lstm", {})
            # map keys
            return {**default, **{k.replace("units_layer1","units_layer1"): v for k,v in lstm_cfg.items()}}
    except Exception:
        return default

CFG = _load_config()
WINDOW = int(os.getenv("LSTM_WINDOW", CFG.get("window_size", 10)))
STEPS = int(os.getenv("LSTM_STEPS", CFG.get("forecast_steps", 30)))
EPOCHS = int(os.getenv("LSTM_EPOCHS", CFG.get("epochs", 10)))
BATCH_SIZE = int(os.getenv("LSTM_BATCH_SIZE", CFG.get("batch_size", 32)))
UNITS1 = int(os.getenv("LSTM_UNITS1", CFG.get("units_layer1", 64)))
UNITS2 = int(os.getenv("LSTM_UNITS2", CFG.get("units_layer2", 32)))

# reproducibility
SEED = int(os.getenv("SEED", 42))
np.random.seed(SEED)
set_seed(SEED)

def _to_supervised(series: np.ndarray, window: int):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    return np.array(X), np.array(y)

def run_lstm_forecast(
    df: pd.DataFrame,
    target_column: str,
    window: int = WINDOW,
    forecast_steps: int = STEPS,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    units1: int = UNITS1,
    units2: int = UNITS2,
    return_series: bool = False
) -> Union[List[float], dict]:
    """
    Trains a small LSTM on the specified target column and returns forecast_steps predictions.
    Returns a list of floats (predicted values). On error, raises ValueError.

    Notes:
    - df must contain a numeric column target_column.
    - For simplicity this uses a single-feature LSTM with MinMax scaling.
    - The function is intentionally conservative (few epochs) to avoid long training times.
    """
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found in dataframe")

    series = df[target_column].dropna().astype(float).values.reshape(-1, 1)
    if len(series) < window + 1:
        raise ValueError(f"Not enough data points for given window={window}. Need at least {window+1} non-null rows.")

    # scaling
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    X, y = _to_supervised(scaled, window)
    # reshape X -> (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # simple model
    model = Sequential()
    model.add(LSTM(units1, return_sequences=True, input_shape=(window, 1)))
    model.add(LSTM(units2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    # early stopping to avoid overfitting / long runs
    es = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True, verbose=0)

    logger.info(f"Training LSTM: window={window}, epochs={epochs}, batch_size={batch_size}")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])

    # forecasting loop
    last_window = scaled[-window:].reshape(1, window, 1)
    preds_scaled = []
    for _ in range(forecast_steps):
        p = model.predict(last_window, verbose=0)
        preds_scaled.append(p[0, 0])
        # slide window
        last_window = np.append(last_window[:, 1:, :], p.reshape(1, 1, 1), axis=1)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten().tolist()
    logger.info(f"Generated {len(preds)} forecast steps for '{target_column}'")

    if return_series:
        return {
            "predictions": preds,
            "units": "same_as_input",
            "model_summary": {
                "window": window,
                "epochs": epochs,
                "batch_size": batch_size,
                "units1": units1,
                "units2": units2
            }
        }
    return preds
