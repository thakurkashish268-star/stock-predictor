import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")


def create_sequences(data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


def train_lstm(model, X_train, y_train, epochs=30, batch_size=32):
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    dataset  = TensorDataset(X_tensor, y_tensor)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss   = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
    return model


def predict_lstm(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        preds    = model(X_tensor).numpy().flatten()
    return preds


def build_xgboost_model():
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )


def train_and_predict(df, forecast_days=30, look_back=60):
    feature_cols = [
        "Close", "Open", "High", "Low", "Volume",
        "MA_7", "MA_21", "MA_50",
        "EMA_12", "EMA_26", "MACD",
        "RSI", "BB_width", "Price_Change", "Log_Volume"
    ]

    data = df[feature_cols].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler.fit(df[["Close"]].values)

    train_size = int(len(scaled_data) * 0.80)
    train_data = scaled_data[:train_size]
    test_data  = scaled_data[train_size - look_back:]

    X_train, y_train = create_sequences(train_data, look_back)
    X_test,  y_test  = create_sequences(test_data,  look_back)

    print("Training LSTM model...")
    model = LSTMModel(input_size=len(feature_cols))
    model = train_lstm(model, X_train, y_train, epochs=30, batch_size=32)

    lstm_train_pred_scaled = predict_lstm(model, X_train)
    lstm_test_pred_scaled  = predict_lstm(model, X_test)

    lstm_train_pred = close_scaler.inverse_transform(lstm_train_pred_scaled.reshape(-1,1)).flatten()
    lstm_test_pred  = close_scaler.inverse_transform(lstm_test_pred_scaled.reshape(-1,1)).flatten()
    y_train_actual  = close_scaler.inverse_transform(y_train.reshape(-1,1)).flatten()
    y_test_actual   = close_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    train_residuals = y_train_actual - lstm_train_pred

    xgb_feature_cols = [
        "MA_7", "MA_21", "MA_50", "EMA_12", "EMA_26",
        "MACD", "RSI", "BB_width", "Price_Change", "Volume_Change", "Log_Volume"
    ]

    xgb_train_features = df[xgb_feature_cols].values[look_back:train_size]
    xgb_test_features  = df[xgb_feature_cols].values[train_size:]

    print("Training XGBoost model...")
    xgb_model = build_xgboost_model()
    xgb_model.fit(xgb_train_features, train_residuals)

    xgb_test_correction = xgb_model.predict(xgb_test_features)

    min_len = min(len(lstm_test_pred), len(xgb_test_correction), len(y_test_actual))
    lstm_test_pred      = lstm_test_pred[:min_len]
    xgb_test_correction = xgb_test_correction[:min_len]
    y_test_actual       = y_test_actual[:min_len]

    hybrid_test_pred = lstm_test_pred + (0.4 * xgb_test_correction)

    mae_lstm    = mean_absolute_error(y_test_actual, lstm_test_pred)
    mae_hybrid  = mean_absolute_error(y_test_actual, hybrid_test_pred)
    rmse_lstm   = np.sqrt(mean_squared_error(y_test_actual, lstm_test_pred))
    rmse_hybrid = np.sqrt(mean_squared_error(y_test_actual, hybrid_test_pred))

    print(f"Forecasting next {forecast_days} days...")
    future_predictions = _forecast_future(
        df, model, xgb_model, close_scaler, scaler,
        look_back, feature_cols, xgb_feature_cols, forecast_days, 0.4
    )

    test_dates   = df.index[train_size:train_size + min_len].tolist()
    last_date    = df.index[-1]
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days
    ).tolist()

    print("Models trained successfully!")

    return {
        "actual":           y_test_actual.tolist(),
        "lstm_pred":        lstm_test_pred.tolist(),
        "hybrid_pred":      hybrid_test_pred.tolist(),
        "test_dates":       [str(d.date()) for d in test_dates],
        "future_dates":     [str(d.date()) for d in future_dates],
        "future_prices":    future_predictions,
        "metrics": {
            "mae_lstm":        round(mae_lstm, 4),
            "mae_hybrid":      round(mae_hybrid, 4),
            "rmse_lstm":       round(rmse_lstm, 4),
            "rmse_hybrid":     round(rmse_hybrid, 4),
            "improvement_pct": round(((mae_lstm - mae_hybrid) / mae_lstm) * 100, 2)
        },
        "train_size":        train_size,
        "historical_dates":  [str(d.date()) for d in df.index.tolist()],
        "historical_close":  df["Close"].tolist()
    }


def _forecast_future(df, model, xgb_model, close_scaler, scaler,
                     look_back, feature_cols, xgb_feature_cols,
                     forecast_days, xgb_weight):

    last_sequence     = scaler.transform(df[feature_cols].values)[-look_back:]
    future_preds      = []
    current_seq       = last_sequence.copy()
    last_xgb_features = df[xgb_feature_cols].values[-1:]

    for _ in range(forecast_days):
        input_tensor     = torch.FloatTensor(current_seq).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            lstm_pred_scaled = model(input_tensor).item()

        lstm_pred      = close_scaler.inverse_transform([[lstm_pred_scaled]])[0][0]
        xgb_correction = xgb_model.predict(last_xgb_features)[0]
        hybrid_price   = lstm_pred + (xgb_weight * xgb_correction)
        future_preds.append(round(float(hybrid_price), 2))

        new_row        = current_seq[-1].copy()
        new_row[0]     = lstm_pred_scaled
        current_seq    = np.vstack([current_seq[1:], new_row])

    return future_preds