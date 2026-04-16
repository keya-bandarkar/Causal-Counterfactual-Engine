import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'processed.csv')
    meta_path = os.path.join(base_dir, 'data', 'feature_meta.json')
    
    df = pd.read_csv(data_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    target_col = meta['target']
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logging.info(f"Training set shape: {X_train.shape}, Target distribution:\n{y_train.value_counts(normalize=True)}")
    
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
    ])
    
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.05, 0.1]
    }
    
    logging.info("Starting GridSearchCV...")
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    logging.info(f"Best parameters: {grid.best_params_}")
    
    best_model = grid.best_estimator_
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_proba)
    logging.info(f"Test ROC-AUC: {auc:.4f}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    if auc < 0.78:
        logging.warning("ROC-AUC is below the target of 0.78, but continuing anyway.")
        
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'xgb_model.joblib')
    joblib.dump(best_model, model_path)
    logging.info(f"Saved model to {model_path}")
    
    # Save test set
    test_df = X_test.copy()
    test_df[target_col] = y_test
    test_path = os.path.join(models_dir, 'test_set.csv')
    test_df.to_csv(test_path, index=False)
    logging.info(f"Saved test set to {test_path}")

if __name__ == '__main__':
    main()
