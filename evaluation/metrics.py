import numpy as np

def proximity(original, cf, feature_types=None):
    dist = 0
    for f in original:
        val_cf = cf.get(f, original[f])
        if feature_types and feature_types.get(f) == 'categorical':
            dist += 1 if str(original[f]) != str(val_cf) else 0
        else:
            dist += abs(float(original[f]) - float(val_cf))
    return dist

def sparsity(original, cf):
    changed = sum(1 for f in original if abs(float(original[f]) - float(cf.get(f, original[f]))) > 1e-5)
    return changed / len(original)

def validity(cf, model_wrapper):
    pred, _ = model_wrapper.predict_single(cf)
    return float(pred == 1)

def causal_validity(cf, original, structural_eq):
    valid_edges = 0
    total_edges = 0
    for parent, children in structural_eq.dag_dict.items():
        if parent in cf and parent in original:
            delta_p = float(cf[parent]) - float(original[parent])
            for child, weight in children.items():
                if child in cf and child in original:
                    total_edges += 1
                    delta_c = float(cf[child]) - float(original[child])
                    if abs(delta_c - weight * delta_p) < 1e-3:
                        valid_edges += 1
    if total_edges == 0:
        return 1.0
    return valid_edges / total_edges

def diversity_score(cf_list, features):
    if len(cf_list) < 2:
        return 0.0
    mat = np.array([[c.get(f, 0) for f in features] for c in cf_list])
    ptp = np.ptp(mat, axis=0)
    ptp[ptp == 0] = 1
    mat = (mat - np.min(mat, axis=0)) / ptp
    
    dist = 0
    pairs = 0
    for i in range(len(mat)):
        for j in range(i+1, len(mat)):
            dist += np.linalg.norm(mat[i] - mat[j])
            pairs += 1
    return dist / pairs if pairs > 0 else 0.0

def feasibility(cf, feature_bounds, mutable_features, original):
    constraints = 0
    satisfied = 0
    
    for f, v in cf.items():
        if f in feature_bounds:
            constraints += 1
            min_v, max_v = feature_bounds[f]
            if min_v <= float(v) <= max_v:
                satisfied += 1
                
    for f in cf:
        if f not in mutable_features:
            constraints += 1
            if abs(float(cf[f]) - float(original[f])) < 1e-5:
                satisfied += 1
                
    return satisfied / constraints if constraints > 0 else 1.0
