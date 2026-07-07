import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(csv_path='heart.csv', test_size=0.2, random_state=42):
    """
    Loads patient health records and partitions them into scalable training matrices.
    """
    # Load raw clinical records
    data = pd.read_csv(csv_path)
    
    # Split features and target column
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Stratified split to preserve representation properties
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize feature metrics for downstream optimal convergence
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reconvert back to DataFrame formats to maintain feature title mappings for interpretability
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    return X_train_df, X_test_df, y_train, y_test
