Heart Disease Prediction & Interpretability Engine
An end-to-end machine learning pipeline engineered to predict cardiovascular disease risk with high accuracy while explicitly resolving the clinical "black box" paradox using local model interpretability frameworks.
The project evaluates five distinct algorithmic paradigms and implements LIME (Local Interpretable Model-agnostic Explanations) to provide clinicians with transparent, feature-weighted diagnostic audit trails.
📂 Repository Structure
heart.csv.xls — The primary clinical dataset containing patient biomarkers (age, cholesterol, resting blood pressure, thalassemia metrics, etc.).
Heart-Disease-LIME Project.pdf — The comprehensive project report detailing the empirical methodology, literature review, and statistical findings.
README.md — Project documentation and setup guide (this file).
📊 Model Performance Hierarchy
The system was cross-validated and tested on a standardized test partition (N=205). Empirical testing revealed a definitive performance hierarchy, validating that tree-based ensemble methods vastly outperform basic linear or proximity-based classifiers on complex physiological data:
Model Tier	Algorithm	Test Accuracy	Key Diagnostic Insights & Trade-offs
Tier 1 (Optimal)	Random Forest	89.98%	Definitive Winner. Achieved a symmetric 90% precision and 90% recall. Suppressed critical false positives and false negatives to just 10 cases each, establishing maximum clinical safety.
Tier 2	Support Vector Classifier (SVC)	82.67%	Strong non-linear mapping using an RBF kernel. High classification recall (87%), but generated a slight inflation of false alarms (22 false positives).
Tier 2	Gaussian Naive Bayes (GNB)	80.00%	The strongest simple baseline. Independent feature calculations successfully reduced false alarms to 26 and captured 88 true positive cases.
Tier 3	Logistic Regression	79.51%	Acceptable baseline tracking, but severely limited by its rigid linear assumptions, yielding a high false-alarm rate (29 false positives).
Tier 4 (Suboptimal)	K-Nearest Neighbors (KNN)	74.80%	Lowest Performer. Highly volatile and sensitive to varying feature scales (e.g., age vs. cholesterol), resulting in an unacceptable 36 false positives.
🔍 Model Interpretability via LIME
To bridge the gap between algorithmic accuracy and clinical accountability, this project implements a LIME pipeline. Instead of presenting a sterile binary output, the system reverse-engineers the high-performing Random Forest model to display an intuitive, feature-weighted explanation for individual patient predictions.
How to Interpret the LIME Visualizations:
Disease Drivers (Orange Bars): Clinical indicators actively pushing the model toward a positive heart disease diagnosis. For high-risk profiles, features such as a defective thalassemia metric (thal <= 2.00) and a complete absence of major colored vessels (ca <= 0.00) provide the highest mathematical weights.
Mitigating Factors (Blue Bars): Clinical metrics arguing against a heart disease diagnosis (such as the absence of typical chest pain cp <= 0.00).
This visualization transforms the machine learning pipeline into a dependable, auditable Clinical Decision Support System (CDSS) that doctors can cross-reference with traditional diagnostics.
🛠️ Environment & Technical Stack
The pipeline is fully optimized to run within a cloud-native infrastructure, completely eliminating local hardware dependencies:
Development Workspace: Google Colab synchronized with Google Drive for automated model persistence.
Data Engineering: pandas, numpy
Machine Learning Suite: scikit-learn (Model Selection, Metrics, Linear Models, Ensemble, SVM, Neighbors)
Explainable AI Framework: lime
🚀 Quick Start & Usage
1. Clone the Repository
Bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2. Install Dependencies
Bash
pip install pandas numpy scikit-learn lime
3. Execution Flow
Load heart.csv.xls into your environment.
Segment features (X) from the target column (y) and apply a 20% test-size partition (random_state=42).
Train the RandomForestClassifier (n_estimators=100).
Initialize the LimeTabularExplainer using your training matrix to audit individual predictions.
