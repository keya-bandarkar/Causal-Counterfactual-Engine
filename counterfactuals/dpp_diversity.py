import numpy as np
import pandas as pd
from dppy.finite_dpps import FiniteDPP

class DPPDiversitySelector:
    def __init__(self, candidates, feature_names):
        self.candidates = candidates
        self.feature_names = feature_names

    def select(self, k=5):
        if len(self.candidates) <= k:
            return self.candidates
            
        matrix = np.array([[c[f] for f in self.feature_names] for c in self.candidates])
        
        # Normalize to handle scale differences
        ptp = np.ptp(matrix, axis=0)
        ptp[ptp == 0] = 1
        norm_matrix = (matrix - np.min(matrix, axis=0)) / ptp
        
        # RBF Kernel
        diffs = norm_matrix[:, np.newaxis, :] - norm_matrix[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=-1)
        gamma = 1.0
        L = np.exp(-gamma * sq_dists)
        
        # Ensure L is symmetric PS
        L = (L + L.T) / 2
        
        dpp = FiniteDPP('likelihood', **{'L': L})
        
        try:
            dpp.sample_exact_k_dpp(size=k)
            indices = dpp.list_of_samples[-1]
            return [self.candidates[i] for i in indices]
        except Exception as e:
            # Fallback to random if exact k-DPP fails
            indices = np.random.choice(len(self.candidates), k, replace=False)
            return [self.candidates[i] for i in indices]
