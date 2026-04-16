import pandas as pd
import numpy as np

class ModelWrapper:
    """Wrapper around trained scikit-learn/imblearn model pipelines."""
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        
    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        else:
            X = X[self.feature_names]
        return self.model.predict(X)
        
    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        else:
            X = X[self.feature_names]
        return self.model.predict_proba(X)
        
    def predict_single(self, row_dict):
        df = pd.DataFrame([row_dict])
        df = df[self.feature_names]
        pred = self.model.predict(df)[0]
        proba = self.model.predict_proba(df)[0]
        if len(proba) > 1:
            proba = float(proba[1])
        else:
            proba = float(proba[0])
        return pred, proba
