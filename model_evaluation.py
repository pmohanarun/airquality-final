import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

print("Loading and preprocessing data...")
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
print("Data preparation completed.")

print("\nTraining Feed Forward Neural Network...")
# Create and train Feed Forward Neural Network
ann = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(n_past, all_data.shape[1])),
    tf.keras.layers.Dense(32, activation='relu'), 
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

ann.compile(loss=tf.keras.losses.Huber(), optimizer='Adam', metrics=['mae','mse'])
ann_history = ann.fit(x_train, y_train, validation_split=0.2, epochs=50, verbose=1)
ann.save('models/ann_model.h5')

print("\nTraining Simple RNN...")
# Create and train Simple RNN
rnn = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(units=64, return_sequences=True, input_shape=(n_past, all_data.shape[1])), 
    tf.keras.layers.SimpleRNN(units=64, return_sequences=True), 
    tf.keras.layers.SimpleRNN(units=32, return_sequences=False), 
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1)
])

rnn.compile(loss=tf.keras.losses.Huber(), optimizer='Adam', metrics=['mae','mse'])
rnn_history = rnn.fit(x_train, y_train, validation_split=0.2, epochs=50, verbose=1)
rnn.save('models/rnn_model.h5')

print("\nTraining LSTM...")
# Create and train LSTM
lstm = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=32, return_sequences=True, input_shape=(n_past, all_data.shape[1])), 
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.LSTM(units=32, return_sequences=False), 
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

lstm.compile(loss=tf.keras.losses.Huber(), optimizer='Adam', metrics=['mae','mse'])
lstm_history = lstm.fit(x_train, y_train, validation_split=0.2, epochs=50, verbose=1)
lstm.save('models/lstm_model.h5')

print("\nTraining CNN-LSTM...")
# Create and train CNN-LSTM
cnn_lstm = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[n_past, all_data.shape[1]]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=16,return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=16, return_sequences=False, dropout=0.25)), 
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(1)
])

cnn_lstm.compile(loss=tf.keras.losses.Huber(), optimizer='Adam', metrics=['mae','mse'])
cnn_lstm_history = cnn_lstm.fit(x_train, y_train, validation_split=0.2, epochs=50, verbose=1)
cnn_lstm.save('models/cnn_lstm_model.h5')

print("\nTraining Attention LSTM...")
# Create and train Attention LSTM
def create_attention_model(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    lstm_out = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    
    # Self-attention mechanism
    attention = tf.keras.layers.Dense(1, activation='tanh')(lstm_out)
    attention = tf.keras.layers.Flatten()(attention)
    attention_weights = tf.keras.layers.Activation('softmax')(attention)
    
    # Apply attention weights
    context_vector = tf.keras.layers.Multiply()([lstm_out, tf.keras.layers.RepeatVector(128)(attention_weights)])
    context_vector = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)
    
    output = tf.keras.layers.Dense(64, activation='relu')(context_vector)
    output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(1)(output)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae','mse'])
    return model

attention_model = create_attention_model((n_past, all_data.shape[1]))
attention_history = attention_model.fit(x_train, y_train, validation_split=0.2, epochs=50, verbose=1)
attention_model.save('models/attention_lstm_model.h5')

print("\nEvaluating all models...")
# Make predictions with all models
models = {
    "Feed Forward Neural Network": ann,
    "Simple RNN": rnn,
    "LSTM": lstm,
    "CNN-LSTM": cnn_lstm,
    "Attention LSTM": attention_model
}

# Evaluate each model
results = []
for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")
    # Make predictions
    predictions = model.predict(x_train[-300:], verbose=0)
    
    # Inverse transform predictions and real values
    predictions = scaler_predict.inverse_transform(predictions)
    real_values = scaler_predict.inverse_transform(y_train[-300:].reshape(-1, 1))
    
    # Calculate metrics
    mae = mean_absolute_error(real_values, predictions)
    mse = mean_squared_error(real_values, predictions)
    rmse = np.sqrt(mse)
    
    results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print("\nModel Evaluation Results:")
print(results_df.to_string())

# Plot the results
plt.figure(figsize=(15, 8))
x = np.arange(len(models))
width = 0.25

plt.bar(x - width, results_df['MAE'], width, label='MAE', color='skyblue')
plt.bar(x, results_df['MSE'], width, label='MSE', color='lightgreen')
plt.bar(x + width, results_df['RMSE'], width, label='RMSE', color='salmon')

plt.xlabel('Models')
plt.ylabel('Error')
plt.title('Model Performance Comparison')
plt.xticks(x, results_df['Model'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

print("\nEvaluation completed. Results have been saved to model_comparison.png") 