import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SCMEstimator:
    def __init__(self, data_df, causal_ordering, feature_bounds=None):
        self.data_df = data_df
        self.causal_ordering = causal_ordering
        self.feature_bounds = feature_bounds
        self.nodes = list(causal_ordering.keys())
        self.adj_matrix = np.zeros((len(self.nodes), len(self.nodes)))
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.idx_to_node = {i: n for i, n in enumerate(self.nodes)}
        
    def estimate_structure(self, prune_threshold=0.1):
        logging.info("Validating DAG skeleton and fitting weights...")
        
        parents_of = {n: [] for n in self.nodes}
        for p, children in self.causal_ordering.items():
            for c in children:
                parents_of[c].append(p)
                
        for child, parents in parents_of.items():
            if not parents:
                continue
            
            X = self.data_df[parents]
            y = self.data_df[child]
            model = LinearRegression().fit(X, y)
            for p, coef in zip(parents, model.coef_):
                if abs(coef) >= prune_threshold:
                    p_idx = self.node_to_idx[p]
                    c_idx = self.node_to_idx[child]
                    self.adj_matrix[p_idx, c_idx] = coef
                else:
                    logging.info(f"Pruned weak edge: {p} -> {child} (weight={coef:.4f})")
                    
        return self.adj_matrix
        
    def visualize(self, output_path):
        G = nx.DiGraph()
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                weight = self.adj_matrix[i, j]
                if weight != 0:
                    G.add_edge(self.idx_to_node[i], self.idx_to_node[j], weight=round(weight, 3))
                    
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_weight='bold', arrows=True)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.title('Discovered Causal DAG')
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Saved DAG visualization to {output_path}")

    def save_dag(self, output_path):
        dag_dict = {}
        for i in range(len(self.nodes)):
            dag_dict[self.idx_to_node[i]] = {}
            for j in range(len(self.nodes)):
                w = self.adj_matrix[i, j]
                if w != 0:
                    dag_dict[self.idx_to_node[i]][self.idx_to_node[j]] = float(w)
        
        with open(output_path, 'w') as f:
            json.dump(dag_dict, f, indent=4)
        logging.info(f"Saved DAG adjacency to {output_path}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'processed.csv')
    meta_path = os.path.join(base_dir, 'data', 'feature_meta.json')
    
    df = pd.read_csv(data_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    estimator = SCMEstimator(df, meta['causal_ordering'])
    estimator.estimate_structure(prune_threshold=0.1)
    
    causal_dir = os.path.join(base_dir, 'causal')
    os.makedirs(causal_dir, exist_ok=True)
    
    estimator.visualize(os.path.join(causal_dir, 'dag.png'))
    estimator.save_dag(os.path.join(causal_dir, 'dag.json'))
    
if __name__ == '__main__':
    main()
