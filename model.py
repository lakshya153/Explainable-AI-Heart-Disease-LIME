from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_ensemble_classifier(X_train, y_train, n_estimators=100, random_state=42):
    """
    Instantiates and optimizes a Random Forest predictive ensemble architecture.
    """
    print("[INFO] Constructing Random Forest Ensemble Model Classifier...")
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_pipeline_model(model, X_test, y_test):
    """
    Computes statistical validation benchmarks across test dimensions.
    """
    y_pred = model.predict(X_test)
    
    metrics_report = classification_report(y_test, y_pred, output_dict=True)
    matrix_res = confusion_matrix(y_test, y_pred)
    
    return metrics_report, matrix_res

def serialize_checkpoint(model, filepath='best_heart_classifier.pkl'):
    """
    Serializes optimal model parameters to an external binary checkpoint.
    """
    joblib.dump(model, filepath)
    print(f"[SUCCESS] Saved model state checkpoints to {filepath}")
