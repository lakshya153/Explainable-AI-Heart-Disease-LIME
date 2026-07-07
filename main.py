import os
import pandas as pd
from preprocess import load_and_preprocess_data
from model import train_ensemble_classifier, evaluate_pipeline_model, serialize_checkpoint
from explain import instantiate_explainability_engine, compute_local_instance_explanation

def run_pipeline():
    # 1. Pipeline initialization and preprocessing
    if not os.path.exists('heart.csv'):
        # Generate generic mock placeholder if dataset isn't loaded locally yet
        raise FileNotFoundError("Please ensure 'heart.csv' is present in the working directory.")

    X_train, X_test, y_train, y_test = load_and_preprocess_data('heart.csv')
    
    # 2. Ensemble Training
    rf_model = train_ensemble_classifier(X_train, y_train)
    
    # 3. Model Quantitative Analysis Evaluation
    report, matrix = evaluate_pipeline_model(rf_model, X_test, y_test)
    print("\n================== PIPELINE VALIDATION METRICS ==================")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
    print("=================================================================\n")
    
    # Save optimized parameters
    serialize_checkpoint(rf_model)
    
    # 4. Local Explanations Framework (XAI Implementation)
    print("\n[INFO] Injecting Explainable AI (XAI) Local Explainer Layer...")
    explainer = instantiate_explainability_engine(X_train)
    
    # Analyze the first index vector row of the test framework
    sample_patient = X_test.iloc[0]
    explanation = compute_local_instance_explanation(explainer, rf_model, sample_patient)
    
    print("\n=== Local Feature Attribution (LIME) Top Factors ===")
    for feature_weight in explanation.as_list():
        print(f"Feature Boundary Condition: {feature_weight[0]} -> Contribution Weight: {feature_weight[1]:.4f}")
    print("====================================================\n")

if __name__ == '__main__':
    run_pipeline()
