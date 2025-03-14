from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('Models/best_model.pkl', 'rb'))
scaler = pickle.load(open('Models/best_scaler.pkl', 'rb'))

# Get feature names from metadata
feature_names = []
with open('Models/model_metadata.txt', 'r') as f:
    for line in f:
        if line.startswith('feature_names:'):
            feature_names = eval(line.split(':', 1)[1].strip())
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    probability = None
    
    if request.method == 'POST':
        # Get form data
        features = {}
        for feature in feature_names:
            features[feature] = float(request.form.get(feature, 0))
        
        # Create DataFrame
        input_df = pd.DataFrame([features])
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Get probability
        probability = model.predict_proba(input_scaled)[0][1]
        probability_percent = round(probability * 100, 2)
        
        result = {
            'prediction': 'Positive' if prediction == 1 else 'Negative',
            'probability': probability_percent,
            'features': features
        }
        
        # Save prediction to history
        save_prediction(features, prediction, probability)
    
    return render_template('predict.html', result=result, feature_names=feature_names)

@app.route('/report')
def report():
    # Load prediction history
    history = []
    if os.path.exists('prediction_history.csv'):
        history = pd.read_csv('prediction_history.csv')
        history = history.to_dict('records')
    
    # Calculate statistics
    stats = calculate_statistics()
    
    return render_template('diabetes-report.html', history=history, stats=stats)

def save_prediction(features, prediction, probability):
    """Save prediction to CSV file"""
    # Create a record
    features['prediction'] = 'Positive' if prediction == 1 else 'Negative'
    features['probability'] = round(probability * 100, 2)
    features['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create DataFrame
    df = pd.DataFrame([features])
    
    # Append to CSV
    if os.path.exists('prediction_history.csv'):
        df.to_csv('prediction_history.csv', mode='a', header=False, index=False)
    else:
        df.to_csv('prediction_history.csv', index=False)

def calculate_statistics():
    """Calculate statistics from prediction history"""
    stats = {
        'total_predictions': 0,
        'positive_count': 0,
        'negative_count': 0,
        'avg_probability': 0,
        'high_risk_count': 0  # probability > 70%
    }
    
    if os.path.exists('prediction_history.csv'):
        df = pd.read_csv('prediction_history.csv')
        stats['total_predictions'] = len(df)
        stats['positive_count'] = len(df[df['prediction'] == 'Positive'])
        stats['negative_count'] = len(df[df['prediction'] == 'Negative'])
        stats['avg_probability'] = round(df['probability'].mean(), 2)
        stats['high_risk_count'] = len(df[df['probability'] > 70])
    
    return stats

if __name__ == '__main__':
    app.run(debug=True)