import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

# ===== 1. DATA LOADING AND PREPROCESSING =====
print("Loading and preprocessing data...")

# Load data
all_data = pd.read_csv("data/input_cluster.csv")

# Handle missing values
all_data = all_data.replace([np.inf, -np.inf], np.nan)
all_data = all_data.fillna(all_data.mean())

# Convert to float
all_data = all_data.astype('float')

print(f"Data shape: {all_data.shape}")
print(f"Columns: {all_data.columns.tolist()}")

# ===== 2. DATA SCALING =====
print("\nScaling data...")

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(all_data)

# Create scaler for predictions
scaler_predict = StandardScaler()
scaler_predict.fit_transform(scaled_data[:, 0:1])

# ===== 3. SEQUENCE CREATION =====
print("\nCreating sequences for time series prediction...")

# Parameters
n_past = 90  # Number of past days to use for prediction
n_future = 60  # Number of future days to predict

# Create sequences
x_train = []
y_train = []

for i in range(n_past, len(scaled_data)):
    x_train.append(scaled_data[i - n_past:i, :])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

# ===== 4. ANN MODEL CREATION AND TRAINING =====
print("\nCreating and training ANN model...")

# Create ANN model
ann = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(n_past, all_data.shape[1])),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile model
ann.compile(loss=tf.keras.losses.Huber(), optimizer='Adam', metrics=['mae', 'mse'])

# Add callbacks for better training
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=15,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5
)

# Train model
print("Training ANN model...")
ann_history = ann.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr]
)

# ===== 5. MODEL EVALUATION =====
print("\nEvaluating ANN model...")

# Make predictions
ann_predictions = ann.predict(x_train[-300:])
ann_predictions = scaler_predict.inverse_transform(ann_predictions)

# Get real values
real_values = scaled_data[-300:, 0]
real_values = scaler_predict.inverse_transform(real_values.reshape(-1, 1))

# Calculate error metrics
ann_mse = mean_squared_error(real_values, ann_predictions)
ann_rmse = np.sqrt(ann_mse)
ann_mae = mean_absolute_error(real_values, ann_predictions)

print(f"ANN MSE: {ann_mse:.2f}")
print(f"ANN RMSE: {ann_rmse:.2f}")
print(f"ANN MAE: {ann_mae:.2f}")

# Calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

ann_mape = mean_absolute_percentage_error(real_values, ann_predictions)
print(f"ANN MAPE: {ann_mape:.2f}%")

# ===== 6. FUTURE PREDICTIONS =====
print("\nMaking future predictions...")

# Make future predictions
future_preds = ann.predict(x_train[-500:])
future_preds = scaler_predict.inverse_transform(future_preds)

# Define date range for predictions
last_day = pd.to_datetime('23-03-2020')
selected_day = pd.to_datetime('20-06-2021')
days_until_today = (selected_day-last_day).days

# Create date list
datelist_future = pd.date_range('04-05-2020', periods=days_until_today, freq='1d').tolist()
datelist_future = pd.to_datetime(datelist_future, dayfirst=True)

# Create DataFrame with predictions
final = pd.DataFrame(data=future_preds[:days_until_today, 0], columns=['AQI'], index=datelist_future)

# Get today's prediction
today = future_preds[-1, 0]
print(f'AQI for the given day is: {today:.2f}')

# Save predictions
final.to_csv('ann_future_predictions.csv')
print("Predictions saved to 'ann_future_predictions.csv'")

# ===== 7. SAVE MODEL =====
print("\nSaving model...")
ann.save('models/ann_model.h5')
print("ANN model saved to 'models/ann_model.h5'")

print("\nAll done!") 