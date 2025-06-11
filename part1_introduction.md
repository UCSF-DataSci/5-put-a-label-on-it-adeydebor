# Install necessary packages

```python
%pip install -r requirements.txt
```
# Part 1: Introduction to Classification & Evaluation

**Objective:** Load the synthetic health data, train a Logistic Regression model, and evaluate its performance.

# 1. Setup - Import necessary libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer

# 2. Data Loading
def load_data(file_path):
    """
    Load the synthetic health data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the data
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found. Please run generate_data.py first.")
        return pd.DataFrame()

# 3. Data Preparation
def prepare_data_part1(df, test_size=0.2, random_state=42):
    """
    Prepare data for modeling: select features, split into train/test sets, handle missing values.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if df.empty:
        return None, None, None, None
    
    # 1. Select relevant features (age, systolic_bp, diastolic_bp, glucose_level, bmi)
    feature_columns = ['age', 'systolic_bp', 'diastolic_bp', 'glucose_level', 'bmi']
    X = df[feature_columns]
    
    # 2. Select target variable (disease_outcome)
    y = df['disease_outcome']
    
    # 3. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 4. Handle missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=feature_columns)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training set disease prevalence: {y_train.mean():.3f}")
    print(f"Test set disease prevalence: {y_test.mean():.3f}")
    
    return X_train, X_test, y_train, y_test

# 4. Model Training
def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained logistic regression model
    """
    if X_train is None or y_train is None:
        return None
    
    # Initialize and train a LogisticRegression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    print("Logistic Regression model trained successfully")
    return model

# 5. Model Evaluation
def calculate_evaluation_metrics(model, X_test, y_test):
    """
    Calculate classification evaluation metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary containing accuracy, precision, recall, f1, auc, and confusion_matrix
    """
    if model is None or X_test is None or y_test is None:
        return {}
    
    # 1. Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 2. Calculate metrics: accuracy, precision, recall, f1, auc
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # 3. Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # 4. Return metrics in a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }
    
    return metrics

# 6. Save Results
def save_results_part1(metrics):
    """Save the calculated metrics to a text file."""
    if not metrics:
        return
    
    # 1. Create 'results' directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # 2. Format metrics as strings and write to file
    with open('results/results_part1.txt', 'w') as f:
        f.write("Part 1: Logistic Regression Results\n")
        f.write("="*40 + "\n")
        f.write(f"accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"precision: {metrics['precision']:.4f}\n")
        f.write(f"recall: {metrics['recall']:.4f}\n")
        f.write(f"f1: {metrics['f1']:.4f}\n")
        f.write(f"auc: {metrics['auc']:.4f}\n")
        
        cm = metrics['confusion_matrix']
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"True Negatives: {cm[0,0]}\n")
        f.write(f"False Positives: {cm[0,1]}\n")
        f.write(f"False Negatives: {cm[1,0]}\n")
        f.write(f"True Positives: {cm[1,1]}\n")
    
    print("Results saved to results/results_part1.txt")

# 8. Interpret Results
def interpret_results(metrics):
    """
    Analyze model performance on imbalanced data.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        
    Returns:
        Dictionary with keys:
        - 'best_metric': Name of the metric that performed best
        - 'worst_metric': Name of the metric that performed worst
        - 'imbalance_impact_score': A score from 0-1 indicating how much
          the class imbalance affected results (0=no impact, 1=severe impact)
    """
    if not metrics:
        return {
            'best_metric': 'unknown',
            'worst_metric': 'unknown',
            'imbalance_impact_score': 0.0
        }
    
    # 1. Determine which metric performed best and worst
    metric_scores = {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'auc': metrics['auc']
    }
    
    best_metric = max(metric_scores, key=metric_scores.get)
    worst_metric = min(metric_scores, key=metric_scores.get)
    
    # 2. Calculate an imbalance impact score based on the difference
    #    between accuracy and more imbalance-sensitive metrics like F1 or recall
    accuracy = metrics['accuracy']
    f1 = metrics['f1']
    recall = metrics['recall']
    
    # Impact score: higher when accuracy >> F1/recall (indicating bias toward majority class)
    f1_diff = abs(accuracy - f1) / max(accuracy, f1) if max(accuracy, f1) > 0 else 0
    recall_diff = abs(accuracy - recall) / max(accuracy, recall) if max(accuracy, recall) > 0 else 0
    imbalance_impact_score = (f1_diff + recall_diff) / 2
    
    # 3. Return the results as a dictionary
    return {
        'best_metric': best_metric,
        'worst_metric': worst_metric,
        'imbalance_impact_score': imbalance_impact_score
    }

# 7. Main Execution
if __name__ == "__main__":
    print("Starting Part 1: Introduction to Classification & Evaluation")
    print("="*60)
    
    # 1. Load data
    data_file = 'data/synthetic_health_data.csv'
    df = load_data(data_file)
    
    if df.empty:
        print("Cannot proceed without data. Please run generate_data.py first.")
    else:
        # 2. Prepare data
        X_train, X_test, y_train, y_test = prepare_data_part1(df)
        
        # 3. Train model
        model = train_logistic_regression(X_train, y_train)
        
        # 4. Evaluate model
        metrics = calculate_evaluation_metrics(model, X_test, y_test)
        
        # 5. Print metrics
        print("\nModel Performance Metrics:")
        print("-" * 30)
        for metric, value in metrics.items():
            if metric != 'confusion_matrix':
                print(f"{metric}: {value:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        # 6. Save results
        save_results_part1(metrics)
        
        # 7. Interpret results
        interpretation = interpret_results(metrics)
        print("\nResults Interpretation:")
        print("-" * 30)
        for key, value in interpretation.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
