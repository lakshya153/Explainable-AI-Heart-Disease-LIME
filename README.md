Heart Disease Prediction & Interpretability Engine
An end-to-end machine learning and data engineering pipeline built to predict cardiovascular disease risk while explicitly resolving the clinical "black box" paradox using local model interpretability frameworks. This system processes clinical biomarkers, evaluates five distinct algorithmic paradigms, and deploys a LIME (Local Interpretable Model-agnostic Explanations) architecture to provide clinicians with transparent, feature-weighted diagnostic audit trails.
Repository Architecture
The project is structured modularly to separate data engineering, modeling, and explainability workflows:
heart.csv.xls: The primary clinical dataset containing patient biomarkers like age, cholesterol, and blood pressure.
preprocess.py: The script managing missing data pipelines, feature scaling, and train-test splits.
model.py: Outlines baseline models including Logistic Regression, KNN, Naive Bayes, and SVC for benchmarking.
main.py: The primary execution script containing the top-performing Random Forest implementation.
explain.py: Handles the integration of the LimeTabularExplainer system for post-hoc interpretations.
Heart-Disease-LIME Project.pdf: The comprehensive engineering report containing full empirical results and literature review.
Model Performance Evaluation
Models were rigorously cross-validated and tested on a standardized validation partition. Testing revealed a definitive performance hierarchy, proving that ensemble tree methods vastly outperform linear or proximity-based classifiers on multi-layered physiological datasets.
The Random Forest Classifier emerged as the optimal architecture, achieving a definitive test accuracy of nearly 90% alongside perfectly balanced precision and recall. It suppressed false positives and false negatives to just 10 cases each, establishing maximum clinical safety.
In comparison, the Support Vector Classifier reached 82.67% accuracy by mapping non-linear boundaries using an RBF kernel, while Gaussian Naive Bayes followed closely at 80.00% by calculating independent features. Logistic Regression marked a baseline at 79.51% but suffered from rigid linear assumptions. K-Nearest Neighbors was the lowest performer at 74.80%, proving highly volatile and sensitive to varying feature scales like age versus cholesterol, which resulted in an unacceptable 36 false positives.
Model Interpretability via LIME
To bridge the gap between algorithmic accuracy and clinical accountability, explainpy implements a LIME pipeline. Instead of presenting a sterile binary output, the system reverse-engineers the complex Random Forest model to generate an intuitive, feature-weighted explanation for individual patient predictions.
Disease Drivers (Orange Bars): Clinical indicators actively pushing the model toward a positive heart disease diagnosis. For high-risk profiles, features such as a defective thalassemia metric and a complete absence of major colored vessels provide the highest mathematical weights.
Mitigating Factors (Blue Bars): Clinical metrics arguing against a heart disease diagnosis, such as the absence of exercise-induced angina.
This visualization transforms the machine learning pipeline into a dependable, auditable Clinical Decision Support System that medical professionals can easily cross-reference.
Quick Start & Usage
1. Install Dependencies
Bash
pip install pandas numpy scikit-learn lime
2. Run the Full Pipeline
To run the pre-processing matrix, evaluate the models, and output the LIME explanation plots, execute:
Bash
python main.py
