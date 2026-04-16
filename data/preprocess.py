import os
import json
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants constraints & ordering
IMMUTABLE_FEATURES = ['age']
MUTABLE_FEATURES = [
    'income', 
    'debt_ratio', 
    'monthly_income',
    'num_open_credit_lines', 
    'num_times_90d_late', 
    'num_real_estate_loans',
    'num_times_60_89d_late', 
    'num_dependents'
]

# DAG skeleton mapping (parent -> [children])
CAUSAL_ORDERING = {
    'age': ['num_dependents', 'income'],
    'income': ['debt_ratio', 'num_open_credit_lines', 'monthly_income'],
    'debt_ratio': [],
    'monthly_income': [],
    'num_open_credit_lines': ['num_times_90d_late', 'num_real_estate_loans', 'num_times_60_89d_late'],
    'num_times_90d_late': [],
    'num_real_estate_loans': [],
    'num_times_60_89d_late': [],
    'num_dependents': []
}

def generate_synthetic_data(n_samples=5000):
    """
    Generate synthetic data resembling the Give Me Some Credit dataset
    """
    logging.info("Generating synthetic dataset (fallback)...")
    np.random.seed(42)
    
    # Generate background variables
    age = np.random.normal(45, 12, n_samples).clip(18, 90).astype(int)
    
    # Causal links
    num_dependents = np.random.poisson(lam=np.where(age < 30, 0.5, np.where(age < 50, 2.0, 0.5))).clip(0, 10).astype(int)
    income = np.random.lognormal(mean=np.log(60000) - np.where(age < 30, 0.5, 0), sigma=0.5, size=n_samples)
    monthly_income = income / 12.0
    
    debt_ratio = np.random.beta(a=2, b=5, size=n_samples) * (150000 / np.maximum(income, 1))
    debt_ratio = np.clip(debt_ratio, 0, 10)
    
    num_open_credit_lines = np.random.poisson(lam=np.log(np.maximum(income, 1000)) * 0.8).clip(1, 30).astype(int)
    
    # More credit lines roughly correlate with real estate loans and late payment chances if income isn't high enough
    num_real_estate_loans = np.random.binomial(n=5, p=np.clip(num_open_credit_lines / 40.0, 0, 1)).astype(int)
    
    late_prob = np.clip(0.1 + 0.1 * debt_ratio - 0.05 * np.log(np.maximum(monthly_income, 1)), 0.01, 0.99)
    num_times_90d_late = np.random.poisson(lam=late_prob * 1.5).astype(int)
    num_times_60_89d_late = np.random.poisson(lam=late_prob * 2.0).astype(int)
    
    # Target definition
    # Higher debt, more lates -> higher risk of default
    logit = -3.0 + 1.2 * num_times_90d_late + 0.8 * num_times_60_89d_late + 0.5 * debt_ratio - 0.0001 * monthly_income
    prob_default = 1 / (1 + np.exp(-logit))
    # Some added noise
    target = np.random.binomial(n=1, p=prob_default)
    
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'debt_ratio': debt_ratio,
        'monthly_income': monthly_income,
        'num_open_credit_lines': num_open_credit_lines,
        'num_times_90d_late': num_times_90d_late,
        'num_real_estate_loans': num_real_estate_loans,
        'num_times_60_89d_late': num_times_60_89d_late,
        'num_dependents': num_dependents,
        'SeriousDlqin2yrs': target
    })
    
    # Introduce some missing dependents randomly
    df.loc[np.random.rand(n_samples) < 0.05, 'num_dependents'] = np.nan
    df.loc[np.random.rand(n_samples) < 0.15, 'monthly_income'] = np.nan
    return df

def clean_and_preprocess(df):
    logging.info("Cleaning and preprocessing data...")
    
    # Median imputation
    for col in df.columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    # IQR method for outliers on specific columns
    for col in ['income', 'debt_ratio']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Instead of dropping, we cap to preserve data matching realistic bounds
        df[col] = df[col].clip(lower=max(0, lower_bound), upper=upper_bound)
        
    # Build bounds
    bounds = {}
    for col in IMMUTABLE_FEATURES + MUTABLE_FEATURES:
        bounds[col] = [float(df[col].min()), float(df[col].max())]
    
    return df, bounds

def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(data_dir, exist_ok=True)
    
    df = generate_synthetic_data(10000)
    df, bounds = clean_and_preprocess(df)
    
    # Save processed data
    out_csv = os.path.join(data_dir, 'processed.csv')
    df.to_csv(out_csv, index=False)
    logging.info(f"Saved processed dataset to {out_csv}")
    
    # Save metadata
    meta = {
        'immutable_features': IMMUTABLE_FEATURES,
        'mutable_features': MUTABLE_FEATURES,
        'feature_bounds': bounds,
        'causal_ordering': CAUSAL_ORDERING,
        'target': 'SeriousDlqin2yrs'
    }
    out_meta = os.path.join(data_dir, 'feature_meta.json')
    with open(out_meta, 'w') as f:
        json.dump(meta, f)
        
    logging.info(f"Saved feature metadata to {out_meta}")

if __name__ == '__main__':
    main()
