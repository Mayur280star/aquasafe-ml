# AquaSafe-ML

Drinking-water potability and safety screening with machine learning.
B.E. SEM-VII Machine Learning mini-project, Rajiv Gandhi Institute of Technology, Mumbai:
Riddhesh V. Budha (A 712), Mayur D. Chaudhari (A 714), Yashodeep Y. Chaudhary (A 715), Hrishita Chavan (A 717).

* **Dataset 1, Potability** (Kaggle water_potability.csv, 9 physicochemical readings): stacking ensemble,
  66.3% test accuracy (baseline 61.0%), 79.2% precision on confident "potable" calls.
* **Dataset 2, Water safety** (Kaggle water-quality, 20 contaminant readings, synthetic): tuned XGBoost,
  97.3% test accuracy (baseline 88.6%), 90.6% precision, F1 0.875. Each sample is tested with all 13 models.

Run locally: `pip install -r requirements.txt && streamlit run streamlit_app.py`

A screening aid only, not a certified water test.
