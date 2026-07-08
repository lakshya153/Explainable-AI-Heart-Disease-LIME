Explainable AI Heart Disease Prediction & Interpretability Engine
An end-to-end machine learning and data engineering pipeline engineered to predict cardiovascular disease risk with high accuracy while explicitly resolving the clinical "black box" paradox using local model interpretability frameworks.
📄 Project Whitepaper
The comprehensive project report detailing the literature review, system architecture, methodology, and performance charts is available directly in this repository: 👉 View Full Technical Report PDF
🔬 Architectural Overview
The system processes clinical patient biomarkers using a decoupled two-stage analysis pipeline:
Stage I (Benchmarking & Performance Optimization): Evaluates a broad spectrum of geometric, probabilistic, linear, and tree-based ensemble classifiers to isolate the optimal diagnostic architecture.
Stage II (Local Interpretability Execution): The high-performing ensemble model is wrapped inside a post-hoc transparency framework to extract local feature weights, explaining the individual physiological catalysts behind a patient's risk profile.
🗂️ Codebase Architecture
This repository is built using a clean, modular pipeline:
main.py — The primary training pipeline loop containing the top-performing Random Forest implementation and final model evaluation outputs.
preprocess.py — Script managing missing data pipelines, outlier mitigation, feature scaling, and train-test splits.
model.py — Outlines baseline models including Logistic Regression, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Support Vector Classifiers (SVC) for architectural benchmarking.
explain.py — Handles the integration of the LimeTabularExplainer system to generate intuitive, feature-weighted explanation plots for clinicians.
heart.csv.xls — The primary clinical dataset containing patient biomarkers (such as age, cholesterol, resting blood pressure, and thalassemia metrics).
📊 Core Performance Metrics
Evaluated using a standardized clinical validation partition, our results demonstrate that ensemble tree models vastly outperform linear or proximity-based classifiers on multi-layered physiological datasets:
Random Forest Classifier: Emerged as the optimal architecture, achieving a definitive test accuracy of nearly 90% alongside perfectly balanced 90% precision and 90% recall. It successfully minimized critical false alarms and missed diagnoses down to just 10 cases each.
Support Vector Classifier (SVC): Reached 82.67% accuracy by mapping non-linear boundaries using an RBF kernel. It maintained high disease recall (87%) but generated an inflation of false alarms (22 false positives).
Gaussian Naive Bayes (GNB): Proved to be a strong simple baseline at 80.00% accuracy, successfully reducing false positives to 26 by treating variables independently.
Logistic Regression: Marked a baseline at 79.51% test accuracy but remained structurally restricted by its rigid linear assumptions, yielding 29 false positives.
K-Nearest Neighbors (KNN): The lowest performing model at 74.80% accuracy. It proved highly sensitive to varying feature scales (e.g., age vs. cholesterol), resulting in an unacceptable 36 false positives.
🛠️ Quickstart Installation
Ensure you have your environment requirements ready:
Bash
pip install pandas numpy scikit-learn lime
To run the pre-processing matrix, execute baseline benchmarks, and output the visual LIME explanation plots, run:
Bash
python main.py
