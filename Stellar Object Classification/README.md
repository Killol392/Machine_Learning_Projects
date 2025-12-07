# Stellar Object Classification (Galaxy vs Star vs QSO)

This project classifies **Galaxies**, **Stars**, and **Quasi-Stellar Objects (QSOs)** using the SDSS *star_classification-1-1-1.csv* dataset (100,000 samples, 18 features). The goal is to build a clean ML pipeline-from preprocessing and outlier handling to model training and tuning-optimized for accuracy, class-balance, and interpretability.

---

## 1. Dataset Summary
- **Size:** 100,000 samples × 18 columns  
- **Target:** `class` (GALAXY, QSO, STAR)  
- **Features:** Photometric magnitudes (`u`, `g`, `r`, `i`, `z`), celestial coordinates (`alpha`, `delta`), survey metadata (`run_ID`, `plate`, `MJD`, etc.)  
- **Class distribution:**  
  - **Galaxy:** 59,445  
  - **QSO:** 18,961  
  - **Star:** 21,594  

---

## 2. Preprocessing
### Key Steps
- Removed constant-value column (`rerun_ID`).  
- Encoded target labels using `LabelEncoder`:
  - GALAXY -> 0  
  - QSO -> 1  
  - STAR -> 2  
- Checked for missing values (none found).  
- Handled placeholder values, skew, and scaling with **StandardScaler**.  
- Used a **stratified sample of 30,000 rows** for efficient model training.  
- Outliers detected via **IQR** and capped to preserve dataset size.

### Outlier Observations
- `redshift` had **8990** outliers -> capped using IQR  
- Photometric bands (`u`, `g`, `r`, `i`, `z`) had moderate outliers  
- Confirmed with Z-score cross-check

---

## 3. Feature Analysis
### Pearson Correlation
Highly correlated photometric bands:
- `i <-> r` (0.96)  
- `z <-> i` (0.97)  
- `r <-> g` (0.93)

Metadata correlations:
- `spec_obj_ID <-> plate` (1.00)  
- `MJD <-> plate` (0.97)

### Feature Importance (Random Forest)
Top contributors:
1. **redshift** (dominant)
2. `z`
3. `g`
4. `i`
5. `u`
6. `spec_obj_ID`
7. `r`

Features with negligible impact: `obj_ID`, `fiber_ID`, `field_ID`

---

## 4. Visual Explorations
- PairGrid and boxplots show strong class separation through **redshift** and moderate separation in photometric bands.  
- QSOs exhibit distinct high-redshift patterns; Stars cluster at near-zero redshift.  
- Galaxies occupy intermediate regions across all features.

---

## 5. Model Training
### Models Evaluated
- Logistic Regression  
- KNN  
- SVM  
- Random Forest  

### Baseline Performance (before tuning)
| Model | Accuracy | Macro-F1 | Notes |
|-------|----------|----------|-------|
| Logistic Regression | 0.61 | 0.32 | Predicts mostly Galaxy |
| SVM | 0.59 | 0.25 | Severe class imbalance |
| KNN | 0.70 | 0.61 | Balanced but limited |
| **Random Forest** | **0.976** | **0.972** | Best performer |

---

## 6. Hyperparameter Tuning (Grid Search)
### Best Parameters Found
**KNN**
- `n_neighbors=9`, `weights='uniform'`, `p=1`  
- CV Score: **0.695**

**Logistic Regression**
- `C=0.01`, `solver='lbfgs'`  
- CV Score: **0.599**

**SVM**
- `C=1`, `kernel='rbf'`, `gamma='scale'`  
- CV Score: **0.595**

**Random Forest**
- `n_estimators=100`, `max_depth=20`, `max_features='sqrt'`  
- CV Score: **0.9754**

### Final Tuned Model Results (Test Set)
**Random Forest (Best Model)**  
- Accuracy: **0.98**  
- Macro-F1: **0.97**  
- STAR: **100% recall**  
- QSO + GALAXY: high precision/recall  
- Confusion matrix shows minimal misclassification  

---

## 7. Key Findings
- **redshift** is the single most informative feature.  
- Photometric bands offer predictable relationships and strong linearity.  
- Traditional linear models fail due to class imbalance and non-linear boundaries.  
- **Random Forest consistently outperforms all models**, even on smaller samples.  
- SVM is computationally expensive and performs poorly on minority classes.

---

## 8. Reflection
This project reinforced the importance of:
- Systematic preprocessing and outlier control  
- Diagnosing class imbalance and feature redundancy  
- Evaluating models beyond accuracy (macro-F1, recall)  
- Choosing models that scale well with data size-Random Forest was efficient, accurate, and robust compared to SVM or Logistic Regression.

Overall, this pipeline demonstrates a complete, production-ready ML workflow for large tabular classification tasks involving astrophysical data.
