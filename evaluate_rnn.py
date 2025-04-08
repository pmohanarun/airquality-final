import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load and preprocess data
all_data = pd.read_csv('data/input_cluster.csv')
all_data = all_data.iloc[:,1:]

# Prepare training data
training_set = all_data.values
scaler = StandardScaler()
training_set_scaled = scaler.fit_transform(training_set)
scaler_predict = StandardScaler()
scaler_predict.fit_transform(training_set[:, 0:1])
train = training_set_scaled

# Create sequences
n_past = 90
x_train = []
y_train = []

for i in range(n_past, len(train)):
    x_train.append(train[i - n_past:i, 0:all_data.shape[1]])
    y_train.append(train[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Create and train Simple RNN
rnn = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(units=64, return_sequences=True, input_shape=(n_past, all_data.shape[1])), 
    tf.keras.layers.SimpleRNN(units=64, return_sequences=True), 
    tf.keras.layers.SimpleRNN(units=32, return_sequences=False), 
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1)
])

rnn.compile(loss=tf.keras.losses.Huber(), optimizer='Adam', metrics=['mae','mse'])
rnn.fit(x_train, y_train, validation_split=0.2, epochs=50, verbose=1)

# Make predictions
predictions = rnn.predict(x_train[-300:], verbose=0)

# Inverse transform predictions and real values
predictions = scaler_predict.inverse_transform(predictions)
real_values = scaler_predict.inverse_transform(y_train[-300:].reshape(-1, 1))

# Calculate metrics
mae = mean_absolute_error(real_values, predictions)
mse = mean_squared_error(real_values, predictions)
rmse = np.sqrt(mse)

print("\nRNN Model Evaluation Results:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}") 