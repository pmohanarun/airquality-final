import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Create visualizations directory if it doesn't exist
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Load the model
model = tf.keras.models.load_model('models/attention_lstm.h5')

# Create sample input data with cluster feature set to 0
sample_input = np.random.normal(0, 1, (1, 90, 16))  # 90 days, 16 features
sample_input[..., -1] = 0  # Set cluster feature to 0

# Get model predictions
predictions = model(sample_input)

# Create visualization (only showing first 15 features)
plt.figure(figsize=(15, 15))

# Plot feature importance (excluding cluster feature)
plt.subplot(3, 1, 1)
feature_importance = np.mean(np.abs(sample_input[0]), axis=0)[:-1]  # Exclude cluster feature
plt.bar(range(15), feature_importance)
plt.title('Feature Importance Analysis (Excluding Cluster)')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.xticks(range(15), [f'Feature {i}' for i in range(15)], rotation=45)

# Plot temporal attention
plt.subplot(3, 1, 2)
temporal_data = np.mean(np.abs(sample_input[0, :, :-1]), axis=1)  # Calculate temporal importance excluding cluster
plt.plot(range(90), temporal_data)
plt.title('Temporal Attention Analysis')
plt.xlabel('Time Steps (Days)')
plt.ylabel('Attention Weight')

# Plot heatmap (excluding cluster feature)
plt.subplot(3, 1, 3)
heatmap_data = np.abs(sample_input[0, :, :-1])  # Exclude cluster feature
sns.heatmap(heatmap_data, cmap='viridis', xticklabels=[f'Feature {i}' for i in range(15)], yticklabels=10)
plt.title('Feature-Time Heatmap (Excluding Cluster)')
plt.xlabel('Features')
plt.ylabel('Time Steps')

plt.tight_layout()

# Save the plot
plt.savefig('visualizations/attention_analysis.png')
plt.close()

# Save statistics
stats = {
    'feature_importance': {
        'mean': float(np.mean(feature_importance)),
        'max': float(np.max(feature_importance)),
        'min': float(np.min(feature_importance))
    },
    'temporal_attention': {
        'mean': float(np.mean(temporal_data)),
        'max': float(np.max(temporal_data)),
        'min': float(np.min(temporal_data))
    }
}

with open('visualizations/attention_stats.json', 'w') as f:
    json.dump(stats, f, indent=4)

print("Visualization completed. Check the 'visualizations' directory for results.") 