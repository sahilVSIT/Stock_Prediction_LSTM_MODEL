import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

#%matplotlib inline
warnings.filterwarnings('ignore')

def import_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', drop=True, inplace=True)
    return df

data_file = "google.csv"  # Replace with data file path
df = import_data(data_file)

def preprocess_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])
    return scaler, scaled_data

scaler, scaled_data = preprocess_data(df)

def create_lstm_model(sequence_length):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(50, activation='relu', return_sequences=True)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

sequence_length = 10  # Adjust this sequence length as needed
model = create_lstm_model(sequence_length)

def split_data(data, sequence_length, train_ratio=0.8):
    train_size = int(len(data) * train_ratio)
    train_data, test_data = data[:train_size], data[train_size:]

    def create_sequences(data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = split_data(scaled_data, sequence_length)

def train_model(model, X_train, y_train, epochs=50, batch_size=64):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

train_model(model, X_train, y_train, epochs=100, batch_size=64)

def make_predictions(model, X_test):
    predicted = model.predict(X_test)
    return predicted

predicted_prices = make_predictions(model, X_test)

def inverse_transform(scaler, predicted_prices, y_test):

    if predicted_prices.shape[-1] != y_test.shape[-1]:
        raise ValueError("Last dimensions of predicted_prices and y_test do not match.")

    predicted_prices = predicted_prices[:, :, 0]

    predicted_prices = scaler.inverse_transform(predicted_prices)

    y_test_inverse = scaler.inverse_transform(y_test)

    return predicted_prices, y_test_inverse


predicted_prices, actual_prices = inverse_transform(scaler, predicted_prices, y_test)

print("predicted_prices shape:", predicted_prices.shape)
print("y_test shape:", y_test.shape)

y_pred = model.predict(X_test)

print("Shape of y_pred:", y_pred.shape)

y_true = y_test  # You may need to reshape y_test if necessary

y_true = y_true[-y_pred.shape[0]:]

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae_per_step = [mean_absolute_error(y_true[:, 0], y_pred[:, i, 0]) for i in range(y_pred.shape[1])]
mse_per_step = [mean_squared_error(y_true[:, 0], y_pred[:, i, 0]) for i in range(y_pred.shape[1])]
rmse_per_step = [np.sqrt(mse) for mse in mse_per_step]

for i in range(y_pred.shape[1]):
    print(f"Time Step {i + 1}:")
    print(f"Mean Absolute Error (MAE): {mae_per_step[i]}")
    print(f"Mean Squared Error (MSE): {mse_per_step[i]}")
    print(f"Root Mean Squared Error (RMSE): {rmse_per_step[i]}")
    print()

mae_per_step = [mean_absolute_error(y_true, y_pred[:, i]) for i in range(y_pred.shape[1])]
mse_per_step = [mean_squared_error(y_true, y_pred[:, i]) for i in range(y_pred.shape[1])]
rmse_per_step = [np.sqrt(mse) for mse in mse_per_step]

plt.plot(y_true, label='Actual Prices', linestyle='-')

for i in range(y_pred.shape[1]):
    plt.plot(y_pred[:, i, 0], label=f'Predicted Prices (Step {i + 1})', linestyle='--')

plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.title('Actual vs. Predicted Prices Over Time')
plt.show()

plt.plot(mae_per_step, label='MAE')
plt.plot(mse_per_step, label='MSE')
plt.plot(rmse_per_step, label='RMSE')

plt.xlabel('Time Step')
plt.ylabel('Error')
plt.legend()
plt.title('Evaluation Metrics Over Time Steps')
plt.show()

time_step_to_visualize = 0  # Replace with the time step you want to visualize

plt.plot(y_true[:, 0], label='Actual Prices', linestyle='-')
plt.plot(y_pred[:, time_step_to_visualize, 0], label=f'Predicted Prices (Step {time_step_to_visualize + 1})', linestyle='--')

plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.title(f'Actual vs. Predicted Prices at Time Step {time_step_to_visualize + 1}')
plt.show()

y_true = y_true.reshape(-1, 1)
y_pred = y_pred.reshape(-1, 1)

num_time_steps = 10

mae_per_step = []
for i in range(num_time_steps):
    start_idx = i * len(y_true) // num_time_steps
    end_idx = (i + 1) * len(y_true) // num_time_steps
    mae_step = mean_absolute_error(y_true[start_idx:end_idx], y_pred[start_idx:end_idx])
    mae_per_step.append(mae_step)
    print(f"MAE for Time Step {i + 1}: {mae_step}")

overall_mae = np.mean(mae_per_step)
print(f"Overall MAE: {overall_mae}")