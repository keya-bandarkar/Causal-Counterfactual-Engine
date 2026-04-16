import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models.predict import ModelWrapper
from causal.structural_equations import StructuralEquations
from counterfactuals.baseline_dice import DiCEBaseline
from counterfactuals.engine import CounterfactualEngine
import evaluation.metrics as metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def plot_radar(df_res, output_path):
    labels = ['Validity', 'Sparsity', 'Proximity (inv)', 'Diversity', 'Causal Validity', 'Feasibility']
    num_vars = len(labels)
    
    baseline = df_res[df_res['Method'] == 'DiCE'].mean(numeric_only=True)
    engine = df_res[df_res['Method'] == 'Ours'].mean(numeric_only=True)
    
    # Proximity is lower=better, invert it for radar
    # Sparsity is lower=better, invert it for radar
    max_prox = max(baseline['Proximity'], engine['Proximity'], 0.1)
    
    def transform(row):
        return [
            row['Validity'],
            1.0 - row['Sparsity'],
            1.0 - (row['Proximity'] / max_prox),
            row['Diversity'],
            row['Causal_Validity'],
            row['Feasibility']
        ]
        
    b_vals = transform(baseline)
    e_vals = transform(engine)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    b_vals += b_vals[:1]
    e_vals += e_vals[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, b_vals, color='red', linewidth=2, label='DiCE Baseline')
    ax.fill(angles, b_vals, color='red', alpha=0.25)
    ax.plot(angles, e_vals, color='blue', linewidth=2, label='Causal Engine')
    ax.fill(angles, e_vals, color='blue', alpha=0.25)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')
            
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Benchmark Comparison', y=1.05)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_path = os.path.join(base_dir, 'models', 'test_set.csv')
    model_path = os.path.join(base_dir, 'models', 'xgb_model.joblib')
    meta_path = os.path.join(base_dir, 'data', 'feature_meta.json')
    dag_path = os.path.join(base_dir, 'causal', 'dag.json')
    
    test_df = pd.read_csv(test_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    with open(dag_path, 'r') as f:
        dag_dict = json.load(f)
        
    target = meta['target']
    features = [c for c in test_df.columns if c != target]
    mutable = meta['mutable_features']
    
    xgb_model = joblib.load(model_path)
    wrapper = ModelWrapper(xgb_model, features)
    
    # Filter 100 rejected
    preds = wrapper.predict(test_df)
    rejected_idx = np.where(preds == 0)[0]
    sample_idx = np.random.RandomState(42).choice(rejected_idx, size=min(100, len(rejected_idx)), replace=False)
    rejected_sample = test_df.iloc[sample_idx]
    
    seq = StructuralEquations(dag_dict, meta['feature_bounds'])
    engine = CounterfactualEngine(test_df, target, wrapper, seq, meta)
    dice = DiCEBaseline(test_df, target, features, wrapper)
    
    results = []
    
    logging.info(f"Running benchmark on {len(rejected_sample)} rejected instances...")
    
    for _, row in rejected_sample.iterrows():
        orig_dict = row.drop(labels=[target]).to_dict()
        
        # DiCE
        dice_cfs = dice.generate(pd.DataFrame([orig_dict]), features_to_vary=mutable, total_CFs=5)
        # Engine
        engine_res = engine.explain(orig_dict, k=5)
        engine_cfs = engine_res['counterfactuals']
        
        methods = [('DiCE', dice_cfs), ('Ours', engine_cfs)]
        for m_name, cfs in methods:
            if not cfs:
                continue
            
            cfs_list = [c['features'] for c in cfs]
            div = metrics.diversity_score(cfs_list, features)
            
            for cf_data in cfs:
                cf = cf_data['features']
                prox = metrics.proximity(orig_dict, cf)
                spars = metrics.sparsity(orig_dict, cf)
                val = metrics.validity(cf, wrapper)
                c_val = metrics.causal_validity(cf, orig_dict, seq)
                feas = metrics.feasibility(cf, meta['feature_bounds'], mutable, orig_dict)
                
                results.append({
                    'Method': m_name,
                    'Validity': val,
                    'Proximity': prox,
                    'Sparsity': spars,
                    'Diversity': div,
                    'Causal_Validity': c_val,
                    'Feasibility': feas
                })
                
    df_res = pd.DataFrame(results)
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    res_path = os.path.join(eval_dir, 'results.csv')
    df_res.to_csv(res_path, index=False)
    
    # Custom summary calculation to avoid pandas string aggregation errors
    summary_data = []
    for m in df_res['Method'].unique():
        sub = df_res[df_res['Method'] == m]
        row_dict = {'Method': m}
        for col in ['Validity', 'Proximity', 'Sparsity', 'Diversity', 'Causal_Validity', 'Feasibility']:
            row_dict[col] = pd.to_numeric(sub[col], errors='coerce').mean()
        summary_data.append(row_dict)
    summary = pd.DataFrame(summary_data)
    logging.info("\n--- Benchmark Summary Table ---")
    logging.info("\n" + summary.to_string(index=False))
    
    plot_radar(df_res, os.path.join(eval_dir, 'comparison_radar.png'))
    logging.info(f"Saved benchmark results and radar chart to {eval_dir}")

if __name__ == '__main__':
    main()
