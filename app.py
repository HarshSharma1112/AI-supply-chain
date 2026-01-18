# app.py - AI Demand Forecasting Model with Flask API
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store trained models
models = {}
scalers = {}

def prepare_features(dates):
    """
    Extract useful features from dates for prediction
    Features: day of week, day of month, month, quarter
    """
    df = pd.DataFrame({'date': pd.to_datetime(dates)})
    
    features = pd.DataFrame({
        'day_of_week': df['date'].dt.dayofweek,  # 0=Monday, 6=Sunday
        'day_of_month': df['date'].dt.day,
        'month': df['date'].dt.month,
        'quarter': df['date'].dt.quarter,
        'year': df['date'].dt.year,
        'days_since_start': (df['date'] - df['date'].min()).dt.days
    })
    
    return features

def train_model(product_id, historical_data):
    """
    Train a Linear Regression model for a specific product
    """
    # Convert to DataFrame
    df = pd.DataFrame(historical_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Prepare features
    X = prepare_features(df['date'])
    y = df['quantity'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Store model and scaler
    models[product_id] = model
    scalers[product_id] = scaler
    
    # Calculate accuracy metrics
    predictions = model.predict(X_scaled)
    mae = np.mean(np.abs(predictions - y))
    rmse = np.sqrt(np.mean((predictions - y) ** 2))
    
    return {
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'data_points': len(df)
    }

def predict_future_demand(product_id, days_ahead, last_date):
    """
    Predict demand for future dates
    """
    if product_id not in models:
        return None
    
    model = models[product_id]
    scaler = scalers[product_id]
    
    # Generate future dates
    last_date = pd.to_datetime(last_date)
    future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
    
    # Prepare features for future dates
    X_future = prepare_features(future_dates)
    X_future_scaled = scaler.transform(X_future)
    
    # Make predictions
    predictions = model.predict(X_future_scaled)
    
    # Ensure predictions are not negative
    predictions = np.maximum(predictions, 0)
    
    return predictions, future_dates

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'message': 'Demand Forecasting AI API is running!',
        'endpoints': {
            '/predict': 'POST - Get demand predictions',
            '/train': 'POST - Train model with historical data',
            '/models': 'GET - List trained models'
        }
    })

@app.route('/train', methods=['POST'])
def train():
    """
    Train a model with historical sales data
    
    Expected JSON format:
    {
        "productId": "P001",
        "historicalData": [
            {"date": "2024-01-01", "quantity": 150},
            {"date": "2024-01-02", "quantity": 165},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'productId' not in data or 'historicalData' not in data:
            return jsonify({
                'error': 'Missing required fields: productId and historicalData'
            }), 400
        
        product_id = data['productId']
        historical_data = data['historicalData']
        
        if len(historical_data) < 5:
            return jsonify({
                'error': 'Need at least 5 data points to train the model'
            }), 400
        
        # Train the model
        metrics = train_model(product_id, historical_data)
        
        return jsonify({
            'message': f'Model trained successfully for product {product_id}',
            'productId': product_id,
            'metrics': metrics
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Training failed',
            'details': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict future demand for a product
    
    Expected JSON format:
    {
        "productId": "P001",
        "daysAhead": 30,
        "historicalData": [
            {"date": "2024-01-01", "quantity": 150},
            {"date": "2024-01-02", "quantity": 165},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'productId' not in data:
            return jsonify({
                'error': 'Missing required field: productId'
            }), 400
        
        product_id = data['productId']
        days_ahead = data.get('daysAhead', 30)
        historical_data = data.get('historicalData', [])
        
        if len(historical_data) == 0:
            return jsonify({
                'error': 'No historical data provided'
            }), 400
        
        # Train model if not already trained
        if product_id not in models:
            if len(historical_data) < 5:
                return jsonify({
                    'error': 'Need at least 5 data points to train the model'
                }), 400
            
            metrics = train_model(product_id, historical_data)
        else:
            metrics = {
                'message': 'Using existing trained model'
            }
        
        # Get last date from historical data
        df = pd.DataFrame(historical_data)
        last_date = pd.to_datetime(df['date']).max()
        
        # Make predictions
        predictions, future_dates = predict_future_demand(
            product_id, 
            days_ahead, 
            last_date
        )
        
        if predictions is None:
            return jsonify({
                'error': 'Model not found. Please train the model first.'
            }), 404
        
        # Calculate statistics
        historical_quantities = df['quantity'].values
        avg_historical = np.mean(historical_quantities)
        avg_predicted = np.mean(predictions)
        
        # Format predictions
        prediction_list = [
            {
                'date': date.strftime('%Y-%m-%d'),
                'predictedDemand': round(float(pred), 2)
            }
            for date, pred in zip(future_dates, predictions)
        ]
        
        return jsonify({
            'message': 'Prediction generated successfully',
            'productId': product_id,
            'daysAhead': days_ahead,
            'predictions': prediction_list,
            'statistics': {
                'avgHistoricalDemand': round(float(avg_historical), 2),
                'avgPredictedDemand': round(float(avg_predicted), 2),
                'minPredicted': round(float(np.min(predictions)), 2),
                'maxPredicted': round(float(np.max(predictions)), 2),
                'totalPredicted': round(float(np.sum(predictions)), 2)
            },
            'modelMetrics': metrics
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List all trained models"""
    return jsonify({
        'message': 'Trained models',
        'models': list(models.keys()),
        'count': len(models)
    })

@app.route('/model/<product_id>', methods=['DELETE'])
def delete_model(product_id):
    """Delete a trained model"""
    if product_id in models:
        del models[product_id]
        del scalers[product_id]
        return jsonify({
            'message': f'Model for {product_id} deleted successfully'
        }), 200
    else:
        return jsonify({
            'error': 'Model not found'
        }), 404

if __name__ == '__main__':
    print("🚀 Starting Demand Forecasting AI Server...")
    print("📊 Machine Learning Model: Linear Regression")
    print("🌐 Server running on http://localhost:5001")
    print("📡 API Endpoints:")
    print("   - POST /predict  : Get demand predictions")
    print("   - POST /train    : Train model with data")
    print("   - GET  /models   : List trained models")
    print("   - DELETE /model/<id> : Delete a model")
    print("\n✅ Ready to accept requests!\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)