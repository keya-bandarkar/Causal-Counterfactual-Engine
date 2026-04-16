import pandas as pd
import numpy as np
import logging
from counterfactuals.milp_generator import MILPCounterfactualGenerator
from counterfactuals.dpp_diversity import DPPDiversitySelector

class CounterfactualEngine:
    def __init__(self, data_df, target_col, model_wrapper, structural_eq, meta):
        self.data_df = data_df
        self.model_wrapper = model_wrapper
        self.structural_eq = structural_eq
        self.meta = meta
        self.target_col = target_col
        self.features = [f for f in data_df.columns if f != target_col]
        
        self.generator = MILPCounterfactualGenerator(data_df, target_col, model_wrapper, structural_eq, meta)
        
    def explain(self, instance_dict, k=5):
        # 1. Check model predicts rejection
        pred, _ = self.model_wrapper.predict_single(instance_dict)
        if pred == 1:
            logging.warning("Instance is already predicting positive class.")
            return {"original": instance_dict, "counterfactuals": [], "metadata": {}}
            
        # 2. Generate N candidates
        candidates = self.generator.generate_seed_candidates(instance_dict, k=k*4)
        
        # 3. Propagate and Filter
        valid_candidates = []
        bounds = self.meta['feature_bounds']
        
        for cand in candidates:
            # We enforce bounding explicitly
            valid = True
            for f, v in cand.items():
                if f in bounds:
                    min_v, max_v = bounds[f]
                    if v < min_v or v > max_v:
                        valid = False
                        break
            
            if valid:
                p, _ = self.model_wrapper.predict_single(cand)
                if p == 1:
                    valid_candidates.append(cand)
                    
        # 4. Diversity Selector
        if len(valid_candidates) == 0:
            logging.warning("MILP failed to find valid CFs, falling back to heuristic causal mutation search.")
            for _ in range(k * 5):
                cand = instance_dict.copy()
                for f in self.meta['mutable_features']:
                    if f in bounds:
                        min_v, max_v = bounds[f]
                        # Move heavily towards typical "good" credit directions implicitly by just uniform sampling
                        cand[f] = np.random.uniform(min_v, max_v)
                
                # Force causal consistency
                if 'income' in cand:
                    cand = self.structural_eq.propagate_intervention('income', cand['income'], cand)
                    
                p, _ = self.model_wrapper.predict_single(cand)
                if p == 1:
                    valid_candidates.append(cand)
                    
            if len(valid_candidates) == 0:
                logging.warning("Heuristic fallback also failed.")
                return {"original": instance_dict, "counterfactuals": [], "metadata": {}}
            
        selector = DPPDiversitySelector(valid_candidates, self.features)
        final_cfs = selector.select(k=min(k, len(valid_candidates)))
        
        # Format the output matching DiCE results
        formatted_cfs = []
        for cand in final_cfs:
            num_changed = sum(1 for f in self.features if abs(cand[f] - instance_dict[f]) > 1e-5)
            proximity = sum(abs(cand[f] - instance_dict[f]) for f in self.features if pd.api.types.is_numeric_dtype(type(cand[f])))
            
            formatted_cfs.append({
                "features": cand,
                "validity": True,
                "proximity": float(proximity),
                "sparsity": num_changed
            })
            
        return {"original": instance_dict, "counterfactuals": formatted_cfs, "metadata": {"pool_size": len(valid_candidates)}}
