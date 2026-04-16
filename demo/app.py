import argparse
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from models.predict import ModelWrapper
from causal.structural_equations import StructuralEquations
from counterfactuals.engine import CounterfactualEngine
import evaluation.metrics as metrics

# Optional UI imports
try:
    import streamlit as st
except ImportError:
    st = None

try:
    import gradio as gr
except ImportError:
    gr = None


def cache_resource_decorator():
    if st is None:
        def identity(func):
            return func
        return identity
    try:
        return st.cache_resource
    except AttributeError:
        return st.cache(allow_output_mutation=True)

cache_resource = cache_resource_decorator()


def load_assets():
    model_path = os.path.join(BASE_DIR, 'models', 'xgb_model.joblib')
    meta_path = os.path.join(BASE_DIR, 'data', 'feature_meta.json')
    dag_path = os.path.join(BASE_DIR, 'causal', 'dag.json')
    test_path = os.path.join(BASE_DIR, 'models', 'test_set.csv')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run models/train.py first.")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Feature metadata not found at {meta_path}.")
    if not os.path.exists(dag_path):
        raise FileNotFoundError(f"DAG file not found at {dag_path}.")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found at {test_path}.")

    xgb_model = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    with open(dag_path, 'r') as f:
        dag_dict = json.load(f)

    test_df = pd.read_csv(test_path)
    features = [c for c in test_df.columns if c != meta['target']]
    wrapper = ModelWrapper(xgb_model, features)
    seq = StructuralEquations(dag_dict, meta['feature_bounds'])
    engine = CounterfactualEngine(test_df, meta['target'], wrapper, seq, meta)

    explainer = shap.TreeExplainer(xgb_model.named_steps['classifier'])
    preds = wrapper.predict(test_df)
    rejected_idx = np.where(preds == 0)[0]
    if len(rejected_idx) > 0:
        default_instance = test_df.iloc[rejected_idx[0]].drop(labels=[meta['target']]).to_dict()
    else:
        default_instance = test_df.iloc[0].drop(labels=[meta['target']]).to_dict()

    return wrapper, engine, seq, meta, explainer, default_instance, features


@cache_resource
def get_assets():
    return load_assets()


def render_streamlit():
    if st is None:
        raise RuntimeError('Streamlit is not available in this environment.')

    wrapper, engine, seq, meta, explainer, default_instance, features = get_assets()
    st.set_page_config(layout='wide', page_title='Causal Counterfactual Engine')
    st.markdown('# Causal Counterfactual Engine')
    st.markdown('Interactive counterfactual analysis for credit decisions using causal structural constraints.')

    st.sidebar.header('Applicant Profile')
    st.sidebar.markdown('Adjust the applicant profile and explore candidate paths to approval.')

    current_instance = {}
    immutable = set(meta.get('immutable_features', []))

    with st.sidebar.expander('Immutable features', expanded=False):
        for f in features:
            if f in immutable:
                st.write(f'**{f}**: {default_instance[f]:.2f}')
                current_instance[f] = default_instance[f]

    with st.sidebar.expander('Mutable features', expanded=True):
        for f in features:
            if f not in immutable:
                min_v, max_v = meta['feature_bounds'][f]
                step = max(0.01, (max_v - min_v) / 100)
                current_instance[f] = st.slider(f, float(min_v), float(max_v), float(default_instance[f]), step=step)

    st.sidebar.markdown('---')
    st.sidebar.subheader('Model settings')
    st.sidebar.write(f'**Target:** {meta["target"]}')
    st.sidebar.write(f'**Engine:** Causal counterfactual search')
    st.sidebar.write(f'**Feature count:** {len(features)}')

    pred, proba = wrapper.predict_single(current_instance)
    status = 'APPROVED' if pred == 1 else 'REJECTED'
    status_color = 'success' if pred == 1 else 'error'

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if pred == 1:
            st.success(f'### Decision: {status}')
        else:
            st.error(f'### Decision: {status}')
        st.write('This result is based on the current profile and the trained model.')

    with col2:
        st.metric(label='Approval Probability', value=f'{proba:.2%}', delta='Higher is better')

    with col3:
        st.metric(label='Decision Status', value=status, delta='Causal counterfactual ready')

    st.markdown('---')
    st.info('Click **Generate Counterfactuals** to compute candidate paths that are more likely to be approved.')

    recalc = st.button('Generate Counterfactuals', type='primary')
    if pred == 1:
        st.success('This applicant is already predicted to be approved. Use the sliders to explore other profiles.')

    if pred == 0 and recalc:
        with st.spinner('Running Causal Engine...'):
            res = engine.explain(current_instance, k=5)
            cfs = res['counterfactuals']

        if not cfs:
            st.warning('No viable counterfactuals were found for this profile.')
        else:
            st.subheader('Recommended Paths to Approval')
            for i, cand_info in enumerate(cfs):
                cf = cand_info['features']
                changes = []
                for f in features:
                    if abs(cf[f] - current_instance[f]) > 1e-5:
                        diff = cf[f] - current_instance[f]
                        direction = 'Increase' if diff > 0 else 'Reduce'
                        changes.append(f'**{direction} {f}** from {current_instance[f]:.2f} to {cf[f]:.2f}')
                c_val = metrics.causal_validity(cf, current_instance, seq)
                validity_icon = '✅' if c_val >= 0.9 else '⚠️'
                with st.expander(f'Path {i+1} {validity_icon}'):
                    st.markdown('### Suggested Changes')
                    st.markdown('• ' + '\n• '.join(changes))
                    st.markdown(f'**Causal Validity Score:** {c_val:.2f}')
                    st.markdown(f'**Sparsity Score:** {cand_info["sparsity"]:.2f}')

            st.markdown('---')
            st.subheader('Current Feature Importance (SHAP)')
            fig, ax = plt.subplots(figsize=(8, 4))
            df_inst = pd.DataFrame([current_instance])[features]
            shap_vals = explainer.shap_values(df_inst)
            shap_vals = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
            y_pos = np.arange(len(features))
            ax.barh(y_pos, shap_vals)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('SHAP Value (Impact on Model Output)')
            plt.tight_layout()
            st.pyplot(fig)
    elif pred == 0:
        st.warning('This profile is rejected. Press Generate Counterfactuals to discover potential paths to approval.')


def gradio_predict(instance_values, features, wrapper):
    instance = {features[i]: instance_values[i] for i in range(len(features))}
    pred, proba = wrapper.predict_single(instance)
    label = 'APPROVED' if pred == 1 else 'REJECTED'
    return label, f'{proba:.2%}'


def launch_gradio():
    if gr is None:
        raise RuntimeError('Gradio is not available in this environment.')

    wrapper, engine, seq, meta, explainer, default_instance, features = get_assets()
    inputs = []
    for f in features:
        min_v, max_v = meta['feature_bounds'][f]
        default = float(default_instance[f])
        inputs.append(gr.Slider(minimum=float(min_v), maximum=float(max_v), step=max(0.01, (max_v - min_v) / 100), label=f, value=default))

    with gr.Blocks() as demo:
        gr.Markdown('# Causal Counterfactual Engine')
        gr.Markdown('Adjust the applicant profile and click Submit to see the model decision.')
        prediction = gr.Label(label='Prediction')
        probability = gr.Textbox(label='Approval Probability')
        submit = gr.Button('Run Prediction')
        submit.click(fn=lambda *vals: gradio_predict(vals, features, wrapper), inputs=inputs, outputs=[prediction, probability])
        gr.Markdown('> Counterfactual generation is only available through the Streamlit interface.')
    demo.launch()


def main():
    parser = argparse.ArgumentParser(description='Launch the counterfactual app.')
    parser.add_argument('--ui', choices=['streamlit', 'gradio'], default=None,
                        help='Choose the UI to run. If omitted, streamlit is preferred.')
    args = parser.parse_args()

    if args.ui == 'gradio':
        launch_gradio()
    elif args.ui == 'streamlit':
        if st is None:
            raise RuntimeError('Streamlit is not installed. Install streamlit or use --ui gradio.')
        render_streamlit()
    else:
        if st is not None:
            render_streamlit()
        elif gr is not None:
            launch_gradio()
        else:
            raise RuntimeError('Install streamlit or gradio to run the demo.')


if __name__ == '__main__':
    main()
