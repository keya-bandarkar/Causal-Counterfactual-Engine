import numpy as np
import pandas as pd
import cvxpy as cp
import logging
from sklearn.linear_model import LogisticRegression

class MILPCounterfactualGenerator:
    def __init__(self, data_df, target_col, model_wrapper, structural_eq, meta):
        self.meta = meta
        self.structural_eq = structural_eq
        self.model_wrapper = model_wrapper
        
        # We need a linear proxy for the XGBoost model to use in CVXPY
        X = data_df.drop(columns=[target_col]).copy()
        # Ensure correct column order
        X = X[model_wrapper.feature_names]
        
        # Get XGBoost predictions to fit our proxy
        xgb_preds = model_wrapper.predict(X)
        self.proxy_model = LogisticRegression(max_iter=1000)
        self.proxy_model.fit(X, xgb_preds)
        
        self.coef = self.proxy_model.coef_[0]
        self.intercept = self.proxy_model.intercept_[0]
        self.features = X.columns.tolist()
        self.feature_to_idx = {f: i for i, f in enumerate(self.features)}

    def generate_seed_candidates(self, instance, k=20):
        orig_x = np.array([instance[f] for f in self.features])
        candidates = []
        bounds = self.meta['feature_bounds']
        immutable = self.meta.get('immutable_features', [])
        
        for i in range(k):
            x = cp.Variable(len(self.features))
            constraints = []
            
            # 1. Bounds constraint
            for f, (min_v, max_v) in bounds.items():
                if f in self.feature_to_idx:
                    idx = self.feature_to_idx[f]
                    constraints.append(x[idx] >= min_v)
                    constraints.append(x[idx] <= max_v)
                
            # 2. Immutability
            for f in immutable:
                if f in self.feature_to_idx:
                    idx = self.feature_to_idx[f]
                    constraints.append(x[idx] == orig_x[idx])
                
            # 3. Validity (linear proxy: w^T x + b >= margin)
            margin = np.random.uniform(0.1, 1.5)
            validity_constraint = cp.sum(cp.multiply(self.coef, x)) + self.intercept >= margin
            
            # 4. Causal constraints (relaxed to soft penalties in objective)
            causal_penalty = 0
            for parent, children in self.structural_eq.dag_dict.items():
                if parent in self.feature_to_idx:
                    p_idx = self.feature_to_idx[parent]
                    for child, weight in children.items():
                        if child in self.feature_to_idx:
                            c_idx = self.feature_to_idx[child]
                            ideal_child_delta = weight * (x[p_idx] - orig_x[p_idx])
                            actual_child_delta = x[c_idx] - orig_x[c_idx]
                            causal_penalty += cp.abs(actual_child_delta - ideal_child_delta)
            
            # Objective: minimize weighted L1 distance + high penalty for breaking causal structural rules
            random_weights = np.random.uniform(0.5, 1.5, len(self.features))
            objective = cp.Minimize(cp.sum(cp.multiply(random_weights, cp.abs(x - orig_x))) + 5.0 * causal_penalty)
            
            prob = cp.Problem(objective, constraints + [validity_constraint])
            try:
                prob.solve(solver=cp.ECOS)
                if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    cf_dict = {f: float(x.value[j]) for j, f in enumerate(self.features)}
                    candidates.append(cf_dict)
            except Exception as e:
                pass
                
        return candidates
