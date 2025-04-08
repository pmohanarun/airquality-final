import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import pickle
from sklearn.cluster import KMeans

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

# ===== 2. LOCATION CLUSTERING =====
print("\nPerforming location clustering...")

# Extract latitude and longitude for clustering
lat_long = all_data[['latitude', 'longitude']].values

# Perform K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
all_data['cluster'] = kmeans.fit_predict(lat_long)

# Save the kmeans model
with open('models/kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
print("K-means model saved to 'models/kmeans.pkl'")

def prepare_cluster_data(cluster_number):
    """
    Prepare data for a specific cluster
    """
    print(f"\nPreparing data for cluster {cluster_number}...")
    
    # Select data for the specific cluster
    cluster_data = all_data[all_data['cluster'] == cluster_number].copy()
    print(f"Number of samples in cluster {cluster_number}: {len(cluster_data)}")
    
    # Scale the data for this cluster
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    scaled_data = pd.DataFrame(scaled_data, columns=cluster_data.columns)
    
    # Create scaler for predictions (only for the target variable)
    scaler_predict = StandardScaler()
    scaler_predict.fit(cluster_data.iloc[:, 0:1])
    
    return scaled_data, scaler_predict

# ===== 3. SEQUENCE CREATION =====
print("\nCreating sequences for time series prediction...")
n_past = 90  # 90 days of training data
n_future = 120  # 120 days of prediction
sequence_length = n_past

# Process each cluster separately
for cluster in range(5):  # We have 5 clusters
    print(f"\nProcessing cluster {cluster}...")
    
    # Get scaled data for this cluster
    scaled_data, scaler_predict = prepare_cluster_data(cluster)
    n_features = scaled_data.shape[1]
    
    x_train = []
    y_train = []
    
    # Convert DataFrame to numpy array for easier indexing
    data_array = scaled_data.values
    
    for i in range(n_past, len(data_array) - n_future + 1):
        x_train.append(data_array[i - n_past:i])
        y_train.append(data_array[i + n_future - 1, 0])  # Target is the first column
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    print(f"Cluster {cluster} - x_train shape: {x_train.shape}")
    print(f"Cluster {cluster} - y_train shape: {y_train.shape}")
    
    # ===== 4. LSTM MODEL CREATION AND TRAINING =====
    print(f"\nCreating and training LSTM model for cluster {cluster}...")
    
    # Create and train LSTM model
    lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(n_past, n_features)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='mse',
                      metrics=['mae', 'mse'])
    
    # Add callbacks for better training
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001
    )
    
    print(f"Training LSTM model for cluster {cluster} (100 epochs)...")
    lstm_model.fit(x_train, y_train,
                   epochs=100,
                   batch_size=40,
                   validation_split=0.2,
                   callbacks=[early_stopping, reduce_lr],
                   verbose=1)
    
    # Save the LSTM model for this cluster
    lstm_model.save(f'models/lstm_90_120_43rmse_cluster_{cluster}.h5')
    print(f"LSTM model for cluster {cluster} saved to 'models/lstm_90_120_43rmse_cluster_{cluster}.h5'")
    
    # ===== 5. ATTENTION LSTM MODEL =====
    print(f"\nCreating and training Attention LSTM model for cluster {cluster}...")
    
    # Create and train Attention LSTM model
    inputs = tf.keras.Input(shape=(n_past, n_features))
    lstm_out = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    attention = tf.keras.layers.Dense(1, activation='tanh')(lstm_out)
    attention = tf.keras.layers.Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax')(attention)
    attention = tf.keras.layers.RepeatVector(64)(attention)
    attention = tf.keras.layers.Permute([2, 1])(attention)
    sent_representation = tf.keras.layers.multiply([lstm_out, attention])
    sent_representation = tf.keras.layers.Lambda(lambda xin: tf.keras.backend.sum(xin, axis=1))(sent_representation)
    outputs = tf.keras.layers.Dense(1)(sent_representation)
    attention_lstm = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    attention_lstm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                          loss='mse',
                          metrics=['mae', 'mse'])
    
    print(f"Training Attention LSTM model for cluster {cluster} (100 epochs)...")
    attention_lstm.fit(x_train, y_train,
                      epochs=100,
                      batch_size=40,
                      validation_split=0.2,
                      callbacks=[early_stopping, reduce_lr],
                      verbose=1)
    
    # Save the Attention LSTM model for this cluster
    attention_lstm.save(f'models/attention_lstm_cluster_{cluster}.h5')
    print(f"Attention LSTM model for cluster {cluster} saved to 'models/attention_lstm_cluster_{cluster}.h5'")
    
    # ===== 6. MODEL EVALUATION =====
    print(f"\nEvaluating models for cluster {cluster}...")
    
    # Define MAPE function
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Get the last 300 actual values (unscaled)
    real_values = all_data[all_data['cluster'] == cluster].iloc[-300:, 0].values.reshape(-1, 1)
    
    # Make predictions with LSTM
    lstm_predictions = lstm_model.predict(x_train[-300:])
    lstm_predictions = scaler_predict.inverse_transform(lstm_predictions)
    
    # Make predictions with Attention LSTM
    attention_predictions = attention_lstm.predict(x_train[-300:])
    attention_predictions = scaler_predict.inverse_transform(attention_predictions)
    
    # Calculate error metrics for LSTM
    lstm_mse = mean_squared_error(real_values, lstm_predictions)
    lstm_rmse = np.sqrt(lstm_mse)
    lstm_mae = mean_absolute_error(real_values, lstm_predictions)
    lstm_mape = mean_absolute_percentage_error(real_values, lstm_predictions)
    
    print(f"\nLSTM Model Metrics for cluster {cluster}:")
    print(f"MSE: {lstm_mse:.4f}")
    print(f"RMSE: {lstm_rmse:.4f}")
    print(f"MAE: {lstm_mae:.4f}")
    print(f"MAPE: {lstm_mape:.2f}%")
    
    # Calculate error metrics for Attention LSTM
    attention_mse = mean_squared_error(real_values, attention_predictions)
    attention_rmse = np.sqrt(attention_mse)
    attention_mae = mean_absolute_error(real_values, attention_predictions)
    attention_mape = mean_absolute_percentage_error(real_values, attention_predictions)
    
    print(f"\nAttention LSTM Model Metrics for cluster {cluster}:")
    print(f"MSE: {attention_mse:.4f}")
    print(f"RMSE: {attention_rmse:.4f}")
    print(f"MAE: {attention_mae:.4f}")
    print(f"MAPE: {attention_mape:.2f}%")
    
    # ===== 7. FUTURE PREDICTIONS =====
    print(f"\nMaking future predictions for cluster {cluster}...")
    
    # Make future predictions with both models
    lstm_future_preds = lstm_model.predict(x_train[-500:])
    attention_future_preds = attention_lstm.predict(x_train[-500:])
    
    # Combine predictions (weighted average)
    combined_preds = 0.6 * attention_future_preds + 0.4 * lstm_future_preds
    future_preds = scaler_predict.inverse_transform(combined_preds)
    
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
    print(f'AQI for cluster {cluster} on the given day is: {today:.2f}')
    
    # Save predictions
    final.to_csv(f'future_predictions_cluster_{cluster}.csv')
    print(f"Predictions for cluster {cluster} saved to 'future_predictions_cluster_{cluster}.csv'")

print("\nAll done!")










