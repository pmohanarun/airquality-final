# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import plotly.graph_objects as go
import plotly.utils
import json
import os
import shutil

# importing prediction input and sequencing it
all_data = pd.read_csv('data/input_cluster.csv')
all_data = all_data.iloc[:,1:]

app = Flask(__name__)

# Create static/plots directory if it doesn't exist
if not os.path.exists('static/plots'):
    os.makedirs('static/plots')

# Copy visualization files to static/plots directory
if os.path.exists('visualizations'):
    for file in os.listdir('visualizations'):
        if file.endswith('.png'):
            shutil.copy2(f'visualizations/{file}', f'static/plots/{file}')

# Load both models for comparison
lstm = tf.keras.models.load_model("models/lstm_90_120_43rmse.h5")
attention_lstm = tf.keras.models.load_model("models/attention_lstm.h5")
kmeans = pickle.load(open('models/kmeans.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/predict')
def predict_page():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/visualization')
def visualization():
    # Load attention statistics
    with open('visualizations/attention_stats.json', 'r') as f:
        stats = json.load(f)
    
    return render_template('visualization.html',
                         feature_stats=stats['feature_importance'],
                         temporal_stats=stats['temporal_attention'])

@app.route('/static/plots/<path:filename>')
def serve_plot(filename):
    return send_from_directory('static/plots', filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the date from the hidden field
        date = request.form.get("date")
        # The date comes in YYYY-MM-DD format
        selected_day = pd.to_datetime(date)
        last_day = pd.to_datetime('2021-04-29')

        # Add validation for past dates
        if selected_day <= last_day:
            return render_template('index.html', 
                error="Cannot predict for past dates. Please select a date after 29-04")

        loc = request.form.get("loc")
        lat_long = [float(x) for x in loc.split(',')]
        
        location = kmeans.predict([lat_long])
        location = int(location)
        
        days_until_today = (selected_day-last_day).days

        # Getting location specific data
        all_data_specific = all_data.loc[all_data.loc[:,'cluster'] == location]
        
        # scaling the selected data
        training_set = all_data_specific.values
        scaler = StandardScaler()
        training_set_scaled = scaler.fit_transform(training_set)
        scaler_predict = StandardScaler()
        scaler_predict.fit_transform(training_set[:, 0:1])
        train = training_set_scaled

        # Prepare data for prediction
        x_train = []
        for i in range(90, len(train)):
            x_train.append(train[i-90:i])
        x_train = np.array(x_train)

        # Make predictions with both models
        preds_lstm = lstm.predict(x_train)
        preds_attention = attention_lstm.predict(x_train)
        
        # Combine predictions (using weighted average)
        preds = 0.6 * preds_attention + 0.4 * preds_lstm
        
        # Create feature importance visualization
        feature_names = ['O3', 'SO2', 'Toluene', 'Benzene', 'CO', 'Xylene', 'Temperature', 'NO2', 'NOx', 'NH3', 'PM10', 'PM2.5']
        feature_importance = np.std(x_train[-1], axis=0)  # Use standard deviation of last sequence
        
        # Remove indices corresponding to Pressure and PO
        indices_to_keep = [i for i, name in enumerate(
            ['O3', 'Pressure', 'SO2', 'Toluene', 'Benzene', 'CO', 'Xylene', 'Temperature', 'NO2', 'NOx', 'NH3', 'PM10', 'PM2.5', 'PO']
        ) if name not in ['Pressure', 'PO']]
        feature_importance = feature_importance[indices_to_keep]
        
        # Create bar plot
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=feature_names,
            y=feature_importance,
            name='Feature Impact',
            marker_color='rgb(66, 133, 244)'  # Added a consistent blue color
        ))
        
        fig.update_layout(
            title='Impact of Air Quality Parameters on AQI Prediction',
            xaxis_title='Air Quality Parameters',
            yaxis_title='Parameter Influence (Higher value = Stronger impact)',
            template='plotly_white',
            height=400,
            showlegend=False  # Remove legend since we only have one trace
        )
        
        # Convert plot to JSON
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get final prediction
        future_predictions = scaler_predict.inverse_transform(preds)[:days_until_today,0]
        today = future_predictions[-1]
        today = round(today,2)
        
        # Determine status and color
        if today <= 50:
            result = "Good"
            color = "rgb(0,117,0)"
            text = "Minimal impact"
        elif 50 < today < 100:
            result = "Satisfactory"
            color = "rgb(126,189,1)"
            text = "May cause minor breathing discomfort to sensitive people"
        elif 100 < today < 200:
            result = "Moderate"
            color = "rgb(242,215,2)"
            text = "May cause breathing discomfort to people with lung disease such as asthma"
        elif 200 < today < 300:
            result = "Poor"
            color = "rgb(244,119,1)"
            text = "May cause breathing discomfort on prolonged exposure"
        elif 300 < today < 400:
            result = "Very poor"
            color = "rgb(218,33,51)"
            text = "May cause respiratory illness on prolonged exposure"
        else:
            result = "Severe"
            color = "rgb(158,25,20)"
            text = "May cause respiratory effects even on healthy people"
        
        return render_template('index.html',
                             prediction=today,
                             result=result,
                             text=text,
                             color=color,
                             graphJSON=graphJSON)
                             
    except Exception as e:
        print(f"Error: {str(e)}")
        return render_template('index.html', error="An error occurred during prediction. Please try again.")

if __name__ == '__main__':
    app.run(debug=True)
