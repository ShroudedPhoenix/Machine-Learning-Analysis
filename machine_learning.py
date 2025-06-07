import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
import joblib


def ML_run():
    data_dir = 'data'
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            # Load the data
            data = pd.read_csv(os.path.join(data_dir, file))

            # Ensure the rows are sorted by the time column
            data = data.sort_values(by='time')

            # Drop duplicates and missing values
            data = data.drop_duplicates().dropna()

            # Define feature columns
            feature_columns = ['open', 'high', 'low', 'close'] + [f'RSI_{i}' for i in range(5, 22)] + \
                              [f'EMA_{i}' for i in range(
                                  50, 251, 10)] + [f'SMA_{i}' for i in range(50, 251, 10)]

            # Ensure the columns exist in the dataset
            feature_columns = [
                col for col in feature_columns if col in data.columns]

            # Initialize scalers
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            # Create the input and output data
            X = scaler_X.fit_transform(data[feature_columns])
            y = scaler_y.fit_transform(data[['close']])

            # Split the data into training and testing sets
            split = int(len(data) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

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
            model.fit(X_train, y_train, epochs=100, batch_size=64,
                      validation_data=(X_test, y_test))

            # Evaluate the model
            loss = model.evaluate(X_test, y_test)
            print(f"Loss for {file}: {loss}")

            # Save the model and scalers
            model_path = os.path.join(model_dir, f"{file[:12]}.h5")
            model.save(model_path)
            joblib.dump(scaler_X, os.path.join(
                model_dir, f"{file[:12]}_scaler_X.pkl"))
            joblib.dump(scaler_y, os.path.join(
                model_dir, f"{file[:12]}_scaler_y.pkl"))

            print(f"Saved model and scalers for {file[:12]}")
