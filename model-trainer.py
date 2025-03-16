import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.impute import KNNImputer, SimpleImputer
from imblearn.over_sampling import SMOTE
import pickle
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Create Models directory if not exists
os.makedirs('Models', exist_ok=True)
os.makedirs('Plots', exist_ok=True)

def load_and_clean_data():
    """Load and thoroughly clean the data"""
    df = pd.read_csv('dataset/diabetes-raw-data.csv')
    
    print("Original data shape:", df.shape)
    print("Summary statistics before cleaning:")
    print(df.describe().round(2))
    
    # Replace zeros with NaN for columns where zero is not biologically possible
    zero_not_possible_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_not_possible_columns:
        df[col] = df[col].replace(0, np.nan)
    
    # Check missing values
    missing_values = df.isnull().sum()
    print("\nMissing values after replacing zeros:")
    print(missing_values)
    
    # Make sure to use SimpleImputer first to handle any NaN values
    # This ensures we don't have NaN values before doing other operations
    initial_imputer = SimpleImputer(strategy='median')
    df_initial = pd.DataFrame(
        initial_imputer.fit_transform(df),
        columns=df.columns
    )
    
    # Now use KNN imputer for more sophisticated imputation
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_initial),
        columns=df_initial.columns
    )
    
    # Verify no NaN values remain
    if df_imputed.isnull().sum().sum() > 0:
        print("WARNING: NaN values still exist after imputation")
        # Force fill any remaining NaNs
        df_imputed = df_imputed.fillna(df_imputed.mean())
    
    # Check for outliers using IQR
    Q1 = df_imputed.quantile(0.25)
    Q3 = df_imputed.quantile(0.75)
    IQR = Q3 - Q1
    
    # Flag outliers
    print("\nOutliers per column:")
    for col in df_imputed.columns:
        if col != 'Outcome':
            outliers = ((df_imputed[col] < (Q1[col] - 1.5 * IQR[col])) | 
                        (df_imputed[col] > (Q3[col] + 1.5 * IQR[col]))).sum()
            print(f"{col}: {outliers} outliers")
    
    # Cap outliers using Winsorization (less aggressive than removing)
    for col in df_imputed.columns:
        if col != 'Outcome':
            lower_bound = Q1[col] - 1.5 * IQR[col]
            upper_bound = Q3[col] + 1.5 * IQR[col]
            df_imputed[col] = np.where(df_imputed[col] < lower_bound, lower_bound, df_imputed[col])
            df_imputed[col] = np.where(df_imputed[col] > upper_bound, upper_bound, df_imputed[col])
    
    # Feature engineering
    # BMI categories
    df_imputed['BMI_Category'] = pd.cut(df_imputed['BMI'], 
                                       bins=[0, 18.5, 25, 30, 100], 
                                       labels=[0, 1, 2, 3])
    
    # Age groups
    df_imputed['Age_Group'] = pd.cut(df_imputed['Age'], 
                                    bins=[20, 30, 40, 50, 100], 
                                    labels=[0, 1, 2, 3])
    
    # Glucose levels
    df_imputed['Glucose_Category'] = pd.cut(df_imputed['Glucose'], 
                                          bins=[0, 70, 100, 126, 200], 
                                          labels=[0, 1, 2, 3])
    
    # Interaction terms
    df_imputed['BMI_x_Age'] = df_imputed['BMI'] * df_imputed['Age']
    df_imputed['Glucose_x_Insulin'] = df_imputed['Glucose'] * df_imputed['Insulin']
    
    # Log transform right-skewed features
    for col in ['Insulin', 'SkinThickness']:
        df_imputed[f'{col}_Log'] = np.log1p(df_imputed[col])
    
    # Create parity feature (has given birth or not)
    df_imputed['Had_Pregnancy'] = (df_imputed['Pregnancies'] > 0).astype(int)
    
    # Convert any potential categorical features to numeric to prevent issues
    for col in df_imputed.columns:
        if df_imputed[col].dtype == 'object' or df_imputed[col].dtype.name == 'category':
            df_imputed[col] = pd.to_numeric(df_imputed[col], errors='coerce')
    
    # Final check for NaN values and fix if any exist
    if df_imputed.isnull().sum().sum() > 0:
        print("WARNING: NaN values detected after feature engineering")
        df_imputed = df_imputed.fillna(df_imputed.mean())
    
    # Plot distributions of key features
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(df_imputed.columns[:8]):
        plt.subplot(2, 4, i+1)
        sns.histplot(data=df_imputed, x=col, hue='Outcome', kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig('Plots/feature_distributions.png')
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    correlation_matrix = df_imputed.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('Plots/correlation_matrix.png')
    
    print("\nProcessed data shape:", df_imputed.shape)
    print("New features added:", [col for col in df_imputed.columns if col not in df.columns])
    
    return df_imputed

def evaluate_model(model, X_test, y_test, model_name, feature_names=None):
    """Comprehensive model evaluation"""
    start_time = time.time()
    
    # Check for NaNs in test data
    if np.isnan(X_test).any():
        print(f"WARNING: NaNs detected in test data before prediction for {model_name}")
        # Replace NaNs with column means
        column_means = np.nanmean(X_test, axis=0)
        inds = np.where(np.isnan(X_test))
        X_test[inds] = np.take(column_means, inds[1])
    
    y_pred = model.predict(X_test)
    
    # Get prediction probabilities if available
    has_proba = hasattr(model, "predict_proba")
    
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1] if has_proba else None
    except:
        print(f"Warning: Could not get prediction probabilities for {model_name}")
        y_pred_proba = None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        pr_auc = average_precision_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    except:
        print(f"Warning: Could not calculate ROC/PR AUC for {model_name}")
        roc_auc = None
        pr_auc = None
    
    inference_time = time.time() - start_time
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    if pr_auc is not None:
        print(f"PR AUC: {pr_auc:.4f}")
    print(f"Inference time: {inference_time:.4f} seconds")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f'Plots/confusion_matrix_{model_name.replace(" ", "_")}.png')
    
    # Feature importance plot for tree-based models
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        plt.figure(figsize=(10, 8))
        importances = pd.DataFrame({'feature': feature_names, 
                                   'importance': model.feature_importances_})
        importances = importances.sort_values('importance', ascending=False)
        sns.barplot(x='importance', y='feature', data=importances[:15])
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(f'Plots/feature_importance_{model_name.replace(" ", "_")}.png')
    
    # Plot ROC curve
    if roc_auc is not None and y_pred_proba is not None:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'Plots/roc_curve_{model_name.replace(" ", "_")}.png')
    
    return {
        'model': model,
        'name': model_name,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'inference_time': inference_time,
        'confusion_matrix': conf_matrix
    }

def train_models():
    """Train multiple models with enhanced processing and evaluation"""
    # Load and clean data with advanced preprocessing
    dataset = load_and_clean_data()
    
    # Feature selection
    X = dataset.drop('Outcome', axis=1)
    y = dataset['Outcome']
    
    # Print dataset info
    print(f"Dataset shape: {X.shape}")
    print(f"Feature columns: {X.columns.tolist()}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Check for NaN values before splitting
    if X.isnull().sum().sum() > 0:
        print("WARNING: NaN values found before train/test split")
        X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Check for class imbalance
    print("\nClass balance in training set:")
    print(y_train.value_counts(normalize=True))
    
    # Apply SMOTE for handling class imbalance
    try:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    except Exception as e:
        print(f"SMOTE failed with error: {e}")
        print("Falling back to original training data")
        X_train_resampled, y_train_resampled = X_train.copy(), y_train.copy()
    
    print("After resampling:")
    print(pd.Series(y_train_resampled).value_counts(normalize=True))
    
    # Feature scaling - try different scalers
    scalers = {
        'MinMaxScaler': MinMaxScaler(),
        'StandardScaler': StandardScaler(),
        'RobustScaler': RobustScaler()  # Better with outliers
    }
    
    # Define models to train including XGBoost
    models = {
        'KNN': KNeighborsClassifier(),
        'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    # Simplified hyperparameter grids to speed up training
    param_grids = {
        'KNN': {
            'n_neighbors': [5, 11, 15],
            'weights': ['uniform', 'distance'],
        },
        'LogisticRegression': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'saga'],
        },
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
        },
        'GradientBoosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        },
        'SVM': {
            'C': [1, 10],
            'kernel': ['rbf', 'linear']
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        }
    }
    
    # Store results
    all_results = []
    best_model_info = None
    best_score = 0
    best_scaler = None
    
    # Use StratifiedKFold for more reliable evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Train and evaluate all models with all scalers
    for scaler_name, scaler in scalers.items():
        print(f"\n{'-'*50}")
        print(f"Using {scaler_name} for feature scaling")
        print(f"{'-'*50}")
        
        # Apply scaler to both original and SMOTE-resampled data
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_test_scaled = scaler.transform(X_test)
        
        # Check for NaN values after scaling
        if np.isnan(X_train_scaled).any() or np.isnan(X_test_scaled).any():
            print(f"WARNING: NaNs detected after scaling with {scaler_name}")
            # Fix NaN values if any exist
            if np.isnan(X_train_scaled).any():
                column_means = np.nanmean(X_train_scaled, axis=0)
                inds = np.where(np.isnan(X_train_scaled))
                X_train_scaled[inds] = np.take(column_means, inds[1])
            
            if np.isnan(X_test_scaled).any():
                column_means = np.nanmean(X_test_scaled, axis=0)
                inds = np.where(np.isnan(X_test_scaled))
                X_test_scaled[inds] = np.take(column_means, inds[1])
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            
            # Grid search for hyperparameter tuning with error handling
            try:
                grid_search = GridSearchCV(
                    model, 
                    param_grids[model_name], 
                    cv=cv, 
                    scoring='roc_auc',
                    n_jobs=-1,
                    error_score=0
                )
                
                # Fit the model using resampled data
                grid_search.fit(X_train_scaled, y_train_resampled)
                
                # Get best model
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                print(f"Best parameters: {best_params}")
                
            except Exception as e:
                print(f"Grid search for {model_name} failed with error: {e}")
                print(f"Falling back to default parameters for {model_name}")
                best_model = model
                best_params = "Default parameters (grid search failed)"
            
            # Evaluate model on original test data
            result = evaluate_model(best_model, X_test_scaled, y_test, f"{model_name} with {scaler_name}", X.columns)
            result['scaler'] = scaler_name
            result['best_params'] = best_params
            
            # Add to results if evaluation succeeded
            if result is not None:
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
    
    if not all_results:
        print("No models were successfully trained and evaluated!")
        return None, None, None
    
    # Sort by ROC AUC, then accuracy for models without ROC AUC
    comparison_table = []
    for result in all_results:
        score = result['roc_auc'] if result['roc_auc'] is not None else result['accuracy']
        comparison_table.append({
            'Model': result['name'],
            'ROC AUC': result['roc_auc'] if result['roc_auc'] is not None else 'N/A',
            'Accuracy': result['accuracy'],
            'Time (s)': result['inference_time']
        })
    
    # Convert to DataFrame for better display
    comparison_df = pd.DataFrame(comparison_table)
    if 'ROC AUC' in comparison_df.columns:
        # Convert 'N/A' strings to NaN for proper sorting
        comparison_df['ROC AUC'] = pd.to_numeric(comparison_df['ROC AUC'], errors='coerce')
        comparison_df = comparison_df.sort_values('ROC AUC', ascending=False, na_position='last')
    else:
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    print(comparison_df.to_string(index=False))
    
    # Save the comparison table
    comparison_df.to_csv('Models/model_comparison.csv', index=False)
    
    # Save the best model and scaler
    print("\n\n" + "="*80)
    print(f"BEST MODEL: {best_model_info['name']}")
    print(f"Parameters: {best_model_info['best_params']}")
    print(f"Accuracy: {best_model_info['accuracy']:.4f}")
    if best_model_info['roc_auc'] is not None:
        print(f"ROC AUC: {best_model_info['roc_auc']:.4f}")
    print(f"Inference time: {best_model_info['inference_time']:.4f} seconds")
    print("="*80)
    
    # Create performance visualization for all models
    plt.figure(figsize=(12, 8))
    
    # Get unique model names without scaler info
    model_names = list(set([result['name'].split(' with ')[0] for result in all_results]))
    
    # Get best results for each model
    best_results = []
    for name in model_names:
        model_results = [r for r in all_results if r['name'].split(' with ')[0] == name]
        best_model_result = max(model_results, 
                               key=lambda x: x['roc_auc'] if x['roc_auc'] is not None else x['accuracy'])
        best_results.append(best_model_result)
    
    # Extract metrics
    names = [r['name'].split(' with ')[0] for r in best_results]
    accuracies = [r['accuracy'] for r in best_results]
    roc_aucs = [r['roc_auc'] if r['roc_auc'] is not None else 0 for r in best_results]
    
    # Create bar chart
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, roc_aucs, width, label='ROC AUC')
    plt.bar(x + width/2, accuracies, width, label='Accuracy')
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Best Model Performance Comparison')
    plt.xticks(x, names, rotation=45)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Plots/model_comparison.png')
    
    # Save best model, scaler and feature list
    try:
        pickle.dump(best_model_info['model'], open('Models/best_model.pkl', 'wb'))
        pickle.dump(best_scaler, open('Models/best_scaler.pkl', 'wb'))
        pickle.dump(X.columns.tolist(), open('Models/feature_names.pkl', 'wb'))
        
        # Save detailed model metadata
        metadata = {
            'model_name': best_model_info['name'],
            'scaler': best_model_info['scaler'],
            'accuracy': best_model_info['accuracy'],
            'roc_auc': best_model_info['roc_auc'],
            'parameters': best_model_info['best_params'],
            'feature_names': X.columns.tolist(),
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'preprocessing': {
                'imputation': 'KNN Imputer + Simple Imputer',
                'outlier_handling': 'Winsorization (IQR method)',
                'class_imbalance': 'SMOTE',
                'feature_engineering': [
                    'BMI_Category', 'Age_Group', 'Glucose_Category', 
                    'BMI_x_Age', 'Glucose_x_Insulin', 'Insulin_Log', 
                    'SkinThickness_Log', 'Had_Pregnancy'
                ]
            }
        }
        
        # Save metadata as JSON
        import json
        with open('Models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Also save as readable text
        with open('Models/model_metadata.txt', 'w') as f:
            for key, value in metadata.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for subkey, subvalue in value.items():
                        f.write(f"  {subkey}: {subvalue}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        print("\nBest model, scaler, and metadata saved to Models/ directory")
        print("Visualizations saved to Plots/ directory")
        
    except Exception as e:
        print(f"Error saving model files: {e}")
    
    return best_model_info['model'], best_scaler, X.columns.tolist()

if __name__ == '__main__':
    train_models()