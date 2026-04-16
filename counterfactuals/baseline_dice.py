import pandas as pd
import numpy as np
import dice_ml
import logging
from dice_ml import Data, Model, Dice

class DiCEBaseline:
    def __init__(self, data_df, target_col, continuous_features, model_obj, backend='sklearn', method='genetic'):
        """
        Initialize the DiCE baseline.
        data_df: the training DataFrame (including target)
        target_col: name of the outcome column
        continuous_features: list of continuous feature names
        model_obj: trained model (needs to have predict/predict_proba like sklearn)
        method: Can be 'random', 'genetic', or 'kdtree' for sklearn models.
                (Gradient-based is theoretically supported here via method='gradient' if passing a deep learning model)
        """
        self.target_col = target_col
        self.continuous_features = continuous_features
        
        self.d_data = Data(dataframe=data_df,
                           continuous_features=self.continuous_features,
                           outcome_name=target_col)
        
        self.d_model = Model(model=model_obj, backend=backend)
        
        try:
            self.exp = Dice(self.d_data, self.d_model, method=method)
        except Exception as e:
            logging.warning(f"DiCE initialization with {method} failed, falling back to 'random'. Error: {e}")
            self.exp = Dice(self.d_data, self.d_model, method='random')

    def generate(self, query_instance, features_to_vary, total_CFs=5):
        """
        Generate counterfactuals for a rejected instance.
        query_instance: A pandas DataFrame containing 1 row (the rejected instance).
        features_to_vary: list of feature names that are allowed to change (MUTABLE).
        
        Returns:
            List of dicts: [{'features': dict, 'validity': bool, 'proximity': float, 'sparsity': int}]
        """
        try:
            dice_exp = self.exp.generate_counterfactuals(
                query_instance, 
                total_CFs=total_CFs, 
                desired_class="opposite",
                features_to_vary=features_to_vary
            )
            
            # The result object contains final_cfs_df if valid CFs were found
            if dice_exp.cf_examples_list[0].final_cfs_df is None:
                return []
                
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df
            orig_dict = query_instance.iloc[0].to_dict()
            results = []
            
            for _, row in cf_df.iterrows():
                cf_dict = row.drop(labels=[self.target_col]).to_dict()
                
                # validity flag
                prediction = row[self.target_col]
                # Assuming 1 is approved and 0 is rejected, different from original target means valid
                valid_flag = (prediction != orig_dict.get(self.target_col, 0))
                
                # Compute raw proximity and sparsity
                num_changed = 0
                proximity = 0.0
                
                for col in cf_dict.keys():
                    cf_val = cf_dict[col]
                    orig_val = orig_dict[col]
                    if pd.api.types.is_numeric_dtype(type(cf_val)):
                        diff = abs(float(cf_val) - float(orig_val))
                        proximity += diff
                        if diff > 1e-5:
                            num_changed += 1
                    else:
                        if cf_val != orig_val:
                            num_changed += 1
                            proximity += 1.0
                            
                results.append({
                    "features": cf_dict,
                    "validity": bool(valid_flag),
                    "proximity": proximity, # Note: this is a simple L1 distance
                    "sparsity": num_changed
                })
            
            return results
        except Exception as e:
            logging.error(f"DiCE generation encountered an error: {e}")
            return []
