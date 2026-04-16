import pandas as pd
import numpy as np

class StructuralEquations:
    def __init__(self, dag_dict, feature_bounds=None):
        """
        dag_dict: {parent: {child: weight}}
        """
        self.dag_dict = dag_dict
        self.feature_bounds = feature_bounds
                
    def propagate_intervention(self, feature, new_value, instance):
        """
        Pearl's 3-step (abduction -> action -> prediction).
        instance: dict of the original feature values.
        """
        cf_instance = instance.copy()
        
        if feature not in cf_instance:
            return cf_instance
            
        old_val = cf_instance[feature]
        delta = new_value - old_val
        cf_instance[feature] = new_value
        
        # BFS traversal for downstream effects
        queue = [(feature, delta)]
        
        while queue:
            current_node, current_delta = queue.pop(0)
            
            children = self.dag_dict.get(current_node, {})
            for child, weight in children.items():
                child_delta = weight * current_delta
                cf_instance[child] += child_delta
                
                # Apply bounds if provided
                if self.feature_bounds and child in self.feature_bounds:
                    min_v, max_v = self.feature_bounds[child]
                    cf_instance[child] = np.clip(cf_instance[child], min_v, max_v)
                    
                queue.append((child, child_delta))
                
        return cf_instance
