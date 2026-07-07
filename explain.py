import numpy as np
from lime.lime_tabular import LimeTabularExplainer

def instantiate_explainability_engine(X_train):
    """
    Initializes a Local Interpretable Model-agnostic Explanations (LIME) model matrix.
    """
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns,
        class_names=['No Disease Risk', 'High Disease Risk'],
        mode='classification'
    )
    return explainer

def compute_local_instance_explanation(explainer, model, data_instance, num_features=5):
    """
    Generates local feature perturbation explanations for an individual clinical sample.
    """
    exp = explainer.explain_instance(
        data_row=data_instance,
        predict_fn=model.predict_proba,
        num_features=num_features
    )
    return exp
