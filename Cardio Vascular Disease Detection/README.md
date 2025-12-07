# Cardiovascular Disease (CVD) Detection – Ensemble ML Pipeline

## Overview
End-to-end ML workflow to predict cardiovascular disease using the **cardio_Heart.csv** dataset (70,000 samples). Covers preprocessing, outlier handling, feature engineering, imbalance treatment, feature selection, baseline models, tuning, calibration, and a stacking ensemble.

## Dataset
- **Samples:** 70,000  
- **Target:** `cardio` (0/1) - almost perfectly balanced  
- **Features:** age, height, weight, systolic/diastolic BP, cholesterol, glucose, smoking, alcohol, activity, gender  
- **Duplicates:** 24 removed  
- **Missing values:** none (median/mode imputation pipeline added)

## Preprocessing
- Dropped `id`; cleaned duplicates  
- Capped outliers (IQR) for continuous features (`age`, `height`, `weight`, `ap_hi`, `ap_lo`)  
- OneHotEncoder for categorical; StandardScaler for numeric  
- **Final transformed features:** 13  
- Train/test split: **80/20**, stratified

## Imbalance Handling
- Nearly balanced, but implemented:  
  - **SMOTE** oversampling  
  - **RandomUnderSampler** undersampling

## Feature Selection (Boruta)
Selected **10 important features**:
`age, height, weight, ap_hi, ap_lo, cholesterol_2, cholesterol_3, gluc_2, gluc_3, active_1`

Used these for all modeling.

## Model Performance (After Tuning)
| Model | Acc | Prec | Rec | F1 | AUC |
|-------|------|--------|--------|---------|--------|
| Logistic Regression | 0.7255 | 0.7526 | 0.6715 | 0.7097 | 0.7919 |
| Decision Tree | 0.7257 | 0.7310 | 0.7138 | 0.7223 | 0.7838 |
| **Random Forest** | **0.7361** | **0.7662** | 0.6792 | 0.7201 | **0.8033** |
| SVM (Linear) | 0.7261 | **0.7760** | 0.6355 | 0.6987 | 0.7922 |
| KNN | 0.7060 | 0.7119 | 0.6916 | 0.7016 | 0.7608 |

**Best overall model → Random Forest (highest Accuracy + AUC)**

## Calibration (Brier Loss)
| Model | Brier |
|--------|---------|
| Logistic Regression | 0.1867 |
| Decision Tree | 0.1892 |
| **Random Forest** | **0.1797** |
| SVM | 0.1866 |
| KNN | 0.2045 |

Random Forest = most reliable probability estimator.

## Three-Layer Stacking Model
**Layer 1:** LR, Decision Tree, KNN, SVM  
**Layer 2:** RF, GradientBoosting, Naive Bayes (meta-learner = Logistic Regression)  
**Layer 3:** MLP + Logistic Regression + Layer-2 Stack (soft voting)

### Final Stacking Results
- **Accuracy:** 0.7324  
- **Precision:** 0.7560  
- **Recall:** 0.6858  
- **F1:** 0.7192  
- **AUC:** 0.8005  
- **Training time:** 640s (heavy)  
- **Prediction time:** 9.8s  

**Insight:** Stacking improved calibration/AUC but gave only marginal accuracy gains vs. Random Forest, with much higher computational cost.

## Summary
A complete ML pipeline for CVD prediction, including:
- Data cleaning + outlier capping  
- Feature scaling + encoding  
- SMOTE/undersampling  
- Boruta feature selection  
- Tuned ML baselines  
- Model calibration  
- Multi-layer stacking ensemble  

**Random Forest emerged as the most efficient and best-performing model overall.**
