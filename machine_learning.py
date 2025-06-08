import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
import joblib


def ML_run():
    data_dir = 'data'
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            # Load the data
            data = pd.read_csv(os.path.join(data_dir, file)).sort_values(
                by='time').drop_duplicates().dropna()

            # Define feature columns
            feature_columns = ['open', 'high', 'low', 'close'] + [f'RSI_{i}' for i in range(5, 22)] + \
                              [f'EMA_{i}' for i in range(
                                  50, 251, 10)] + [f'SMA_{i}' for i in range(50, 251, 10)]

            # Ensure the columns exist in the dataset
            feature_columns = [
                col for col in feature_columns if col in data.columns]

            data['target'] = data['close'].shift(-1)
            data = data.dropna(subset=['target'])

            split = int(len(data) * 0.8)
            X_train_raw, X_test_raw = data[feature_columns][:
                                                            split], data[feature_columns][split:]
            y_train_raw, y_test_raw = data[[
                'target']][:split],      data[['target']][split:]

            # Initialize scalers
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            # Split the data into training and testing sets
            X_train = scaler_X.fit_transform(X_train_raw)
            X_test = scaler_X.transform(X_test_raw)
            y_train = scaler_y.fit_transform(y_train_raw)
            y_test = scaler_y.transform(y_test_raw)

            # Create the model
            model = Sequential()
            model.add(Dense(128, input_dim=len(
                feature_columns), activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='linear'))

            # Compile the model
            model.compile(loss='mean_squared_error',
                          optimizer=Adam(learning_rate=0.0005))

            # Train the model
            es = EarlyStopping(patience=10, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=100, batch_size=64,
                      validation_data=(X_test, y_test), callbacks=[es])

            # Evaluate the model
            loss = model.evaluate(X_test, y_test, verbose=0)
            print(f"Loss for {file}: {loss}")

            # Save the model and scalers
            tag = os.path.splitext(file)[0][:12]          # first 12 chars
            model.save(os.path.join(model_dir, f'{tag}.h5'))
            joblib.dump(scaler_X, os.path.join(
                model_dir, f'{tag}_scaler_X.pkl'))
            joblib.dump(scaler_y, os.path.join(
                model_dir, f'{tag}_scaler_y.pkl'))
            print(f'Saved model and scalers for {tag}')
