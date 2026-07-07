# Clinical Diagnostics: Heart Disease Prediction using Transparent Machine Learning Frameworks

An end-to-end predictive learning pipeline utilizing ensemble architectures to forecast cardiovascular risk matrices paired with Local Interpretable Model-agnostic Explanations (LIME) to enforce clinical trust validation.

## 📄 Project Technical Document
The complete engineering architecture detailing the research domain, statistical constraints, schema specifications, and evaluation frameworks is accessible here:
👉 **[View Full Document Whitepaper](./Heart-Disease-Lime%20Project.pdf)**

---

## 🔬 Explainable AI (XAI) Overview
Traditional medical classifiers often act as mathematical "black boxes," leaving clinical decision-makers blind to underlying correlation factors. This repository addresses this constraint by decoupling prediction generation from attribution mappings:
1. **Predictive Ensemble:** Optimized Multi-Tree Random Forest models analyze non-linear patient physiological interactions (lipid configurations, vascular pressure limits, demographic profiles).
2. **Interpretability Architecture:** LIME engines dynamically perturb local features around unique patient vectors to measure variance shifts, establishing transparent boundary weights explaining exactly *why* a risk classification was determined.

---

## 🤖 Why LIME? (Local Interpretable Model-agnostic Explanations)

While the Random Forest ensemble model delivers high predictive accuracy, it operates as a "black box." To ensure clinical safety and transparency, this project implements **LIME** to break down individual patient risk assessments:

* **Local Fidelity:** For any single patient instance, LIME perturbs the clinical data features to see how the model's predictions shift.
* **Feature Weights:** It learns an interpretable linear model around that specific patient, outputting exact contribution scores for critical features like `age`, `cholesterol`, `max heart rate (thalach)`, and `chest pain type (cp)`.
* **Clinical Trust:** This transforms a simple "High Risk" alert into an actionable medical explanation (e.g., *"Patient flagged as High Risk primarily driven by ST depression > 2.0 and Reversible Defect Thalassemia"*).

---

## 🗂️ Production Layout

This codebase features clean modular script packaging:
* `preprocess.py`: Manages vector parsing, feature data scaling (StandardScaler matrices), and stratified training/validation dataset generation.
* `model.py`: Instantiates Random Forest ensemble objects, runs state training loops, evaluates classification report metrics, and serializes state dictionary binary checkpoints.
* `explain.py`: Hosts the tabular LIME engine wrapper generating local patient risk parameter explanations.
* `main.py`: The master execution script orchestrating the dataset pipeline.

---

## 🛠️ Quickstart

Install production library prerequisites:
```bash
pip install pandas numpy scikit-learn lime joblib
