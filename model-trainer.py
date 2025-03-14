import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import pickle
import os
import time

# Create Models directory if not exists
os.makedirs('Models', exist_ok=True)

def load_data():
    """Load and preprocess data"""
    df = pd.read_csv('dataset/diabetes-raw-data.csv')
    
    # Replace zeros with NaN
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[zero_columns] = df[zero_columns].replace(0, np.nan)
    
    # Fill missing values with mean
    df.fillna(df.mean(), inplace=True)
    
    return df

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate ROC AUC if probabilities are available
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    inference_time = time.time() - start_time
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Inference time: {inference_time:.4f} seconds")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return {
        'model': model,
        'name': model_name,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'inference_time': inference_time
    }

def train_models():
    """Train multiple models and select the best one"""
    # Load and preprocess data
    dataset = load_data()
    
    # Feature selection (using all features now)
    X = dataset.drop('Outcome', axis=1)
    y = dataset['Outcome']
    
    # Print dataset info
    print(f"Dataset shape: {X.shape}")
    print(f"Feature columns: {X.columns.tolist()}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Feature scaling - try both scalers
    scalers = {
        'MinMaxScaler': MinMaxScaler(),
        'StandardScaler': StandardScaler()
    }
    
    # Define models to train
    models = {
        'KNN': KNeighborsClassifier(),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # Hyperparameter grids
    param_grids = {
        'KNN': {'n_neighbors': [5, 10, 15, 20, 24], 'weights': ['uniform', 'distance']},
        'LogisticRegression': {'C': [0.1, 1, 10], 'solver': ['liblinear', 'saga']},
        'RandomForest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
        'GradientBoosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
    }
    
    # Store results
    all_results = []
    best_model_info = None
    best_score = 0
    best_scaler = None
    
    # Train and evaluate all models with both scalers
    for scaler_name, scaler in scalers.items():
        print(f"\n{'-'*50}")
        print(f"Using {scaler_name} for feature scaling")
        print(f"{'-'*50}")
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            
            # Grid search for hyperparameter tuning
            grid_search = GridSearchCV(
                model, 
                param_grids[model_name], 
                cv=5, 
                scoring='roc_auc',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            
            # Evaluate model
            result = evaluate_model(best_model, X_test_scaled, y_test, model_name)
            result['scaler'] = scaler_name
            result['best_params'] = grid_search.best_params_
            all_results.append(result)
            
            # Update best model if better
            score_metric = result['roc_auc'] if result['roc_auc'] is not None else result['accuracy']
            if score_metric > best_score:
                best_score = score_metric
                best_model_info = result
                best_scaler = scaler
    
    # Print summary of all models
    print("\n\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    for result in sorted(all_results, key=lambda x: x['roc_auc'] if x['roc_auc'] is not None else x['accuracy'], reverse=True):
        score = result['roc_auc'] if result['roc_auc'] is not None else result['accuracy']
        print(f"{result['name']} with {result['scaler']}: Score = {score:.4f}, Time = {result['inference_time']:.4f}s")
    
    # Save the best model and scaler
    print("\n\n" + "="*80)
    print(f"BEST MODEL: {best_model_info['name']} with {best_model_info['scaler']}")
    print(f"Parameters: {best_model_info['best_params']}")
    print(f"Accuracy: {best_model_info['accuracy']:.4f}")
    if best_model_info['roc_auc'] is not None:
        print(f"ROC AUC: {best_model_info['roc_auc']:.4f}")
    print(f"Inference time: {best_model_info['inference_time']:.4f} seconds")
    print("="*80)
    
    # Save best model and scaler
    pickle.dump(best_model_info['model'], open('Models/best_model.pkl', 'wb'))
    pickle.dump(best_scaler, open('Models/best_scaler.pkl', 'wb'))
    
    # Save model metadata
    metadata = {
        'model_name': best_model_info['name'],
        'scaler': best_model_info['scaler'],
        'accuracy': best_model_info['accuracy'],
        'roc_auc': best_model_info['roc_auc'],
        'parameters': best_model_info['best_params'],
        'feature_names': X.columns.tolist()
    }
    
    with open('Models/model_metadata.txt', 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print("\nBest model, scaler, and metadata saved to Models/ directory")

if __name__ == '__main__':
    train_models()