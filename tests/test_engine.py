import pytest
from evaluation.metrics import causal_validity
from causal.structural_equations import StructuralEquations

def test_causal_validity_metric():
    dag_dict = {'A': {'B': 2.0}, 'B': {'C': -1.0}}
    seq = StructuralEquations(dag_dict, {})
    orig = {'A': 10, 'B': 5, 'C': 5}
    
    # valid CF: change A by +1 -> B should change by +2 -> C should change by -2
    cf1 = {'A': 11, 'B': 7, 'C': 3}
    assert causal_validity(cf1, orig, seq) == 1.0
    
    # invalid CF: C didn't change correctly
    cf2 = {'A': 11, 'B': 7, 'C': 5}
    assert causal_validity(cf2, orig, seq) == 0.5 # 1 out of 2 edges respected
