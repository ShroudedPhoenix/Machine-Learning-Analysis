import os
import joblib
import pandas as pd
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore


import_dir = 'import'
model_dir = 'models'
data_dir = 'data'


def run_predictions():
    for file in os.listdir(data_dir):
        name = file[:12]

        model_path = os.path.join(model_dir, f"{name}.h5")
        scaler_X_path = os.path.join(model_dir, f"{name}_scaler_X.pkl")
        scaler_y_path = os.path.join(model_dir, f"{name}_scaler_y.pkl")

        if not (os.path.exists(model_path) and os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path)):
            print(f"Skipping {file}: Model or scaler files not found.")
            continue

        model = load_model(model_path)
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(learning_rate=0.0005))
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)

        matched_file = next((f for f in os.listdir(
            data_dir) if f.startswith(name)), None)

        if matched_file:
            data_path = os.path.join(data_dir, matched_file)
            data = pd.read_csv(data_path)

            # Check if 'time' column exists before dropping
            if 'time' in data.columns:
                data = data.drop(columns=['time'])

            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close'] + [f'RSI_{i}' for i in range(5, 22)] + \
                            [f'EMA_{i}' for i in range(
                                50, 251, 10)] + [f'SMA_{i}' for i in range(50, 251, 10)]

            if not all(col in data.columns for col in required_cols):
                print(
                    f"Skipping {file}: Missing required columns in data file.")
                continue

            # Keep only the last row
            data = data.tail(1)

            # Scale input features
            scaled_input = scaler_X.transform(data[required_cols])

            # Make prediction
            scaled_prediction = model.predict(scaled_input)
            prediction = scaler_y.inverse_transform(
                scaled_prediction.reshape(-1, 1))

            print(f"Prediction for {file}: {prediction[0][0]}")
        else:
            print(f"No matching data file found for {file}.")
