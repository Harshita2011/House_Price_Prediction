# Lab 01: House Price Prediction using Linear Regression

**Registration Number:** 23MID0043  
**Course:** MDI3003 – Advanced Predictive Analytics  
**Lab:** House Price Prediction using Linear Regression  
**Date:** July 2026

---

## 1. Project Overview

This repository contains an end-to-end regression study comparing classical linear models, regularized linear models, tree-based models, and ensemble methods on three real-world house-price datasets:

1. **Ames Housing** (Iowa, USA) – sale price prediction with mixed numerical and categorical features.
2. **California Housing** – median district house value prediction from census features.
3. **UCI Real Estate Valuation** (Taiwan) – house price per unit area prediction.

The objective is to develop a reproducible, leakage-safe pipeline, establish transparent baselines, compare at least five regression models, and document findings in an industry-style technical report.

---

## 2. Repository Structure

```
Advanced_Predictive_Analytics_LAB/
└── LAB_01/
    ├── 23MID0043_Lab01_Ames_Housing.ipynb          # Ames Housing analysis notebook
    ├── 23MID0043_Lab01_California.ipynb            # California Housing analysis notebook
    ├── 23MID0043_Lab01_UCI_RealEstate.ipynb        # UCI Real Estate analysis notebook
    ├── 23MID0043_Lab01_Ames_Housing.py             # Ames Housing script (generated from notebook)
    ├── 23MID0043_Lab01_California.py               # California Housing script (generated from notebook)
    ├── 23MID0043_Lab01_UCI_RealEstate.py           # UCI Real Estate script (generated from notebook)
    ├── 23MID0043_Lab01_README.md                   # This file
    ├── 23MID0043_Lab01_Report.pdf                  # Industry-style technical report
    ├── requirements.txt                            # Python package dependencies
    ├── execute_notebooks.py                        # Notebook runner script
    ├── train.csv                                   # Ames Housing raw training data
    ├── ames_model_comparison.csv                   # Ames test-set model comparison
    ├── ames_cross_validation.csv                   # Ames 5-fold CV results
    ├── ames_house_price_pipeline.joblib            # Final Ames pipeline artifact
    ├── California_model_comparison.csv             # California test-set model comparison
    ├── California_cross_validation_results.csv     # California 5-fold CV results
    ├── California_house_price_pipeline.joblib      # Final California pipeline artifact
    ├── uci_model_comparison.csv                    # UCI test-set model comparison
    ├── uci_cross_validation.csv                    # UCI 5-fold CV results
    ├── uci_real_estate_model.joblib                # Final UCI pipeline artifact
    └── run_metadata.json                           # Run metadata (seed, sizes, selected model)
```

---

## 3. Environment

- **Python:** 3.11+
- **Key packages:**
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `joblib`
  - `ucimlrepo`
  - `statsmodels`

Install dependencies with:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib ucimlrepo statsmodels
```

All notebooks use a fixed random seed (`SEED = 42`) for reproducibility.

---

## 4. Execution Order

1. **UCI Real Estate notebook** (`23MID0043_Lab01_UCI_RealEstate.ipynb`)  
   Small, clean baseline dataset with no missing values. Good starting point for understanding the workflow.

2. **California Housing notebook** (`23MID0043_Lab01_California.ipynb`)  
   Larger numeric dataset useful for studying scale, geography, and nonlinear patterns.

3. **Ames Housing notebook** (`23MID0043_Lab01_Ames_Housing.ipynb`)  
   High-dimensional mixed-type dataset requiring full preprocessing, categorical encoding, and feature engineering.

Each notebook is self-contained and can be run independently. Run cells sequentially from top to bottom after a fresh kernel restart.

---

## 5. Modeling Workflow

Each notebook follows the same reproducible pipeline:

1. **Business framing** – define target, observation unit, and decision context.
2. **Data audit** – schema, missing values, duplicates, target distribution.
3. **EDA** – correlation heatmap, feature-target scatterplots, categorical boxplots.
4. **Train-test split** – 80/20 random holdout with `random_state=42`.
5. **Leakage-safe preprocessing** – imputation, one-hot encoding, standard scaling inside `ColumnTransformer` and `Pipeline`.
6. **Baselines** – dummy mean predictor and simple Linear Regression.
7. **Model comparison** – Linear, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Gradient Boosting.
8. **Cross-validation** – 5-fold CV on training data.
9. **Hyperparameter tuning** – `GridSearchCV` on Ridge (alpha grid).
10. **Residual diagnostics** – residual plots and actual-vs-predicted plots.
11. **Feature importance / coefficients** – linear coefficients and tree-based importance.
12. **Artifact saving** – CSVs and final `joblib` pipeline.

---

## 6. Evaluation Metrics

- **MAE** – Mean Absolute Error (original target units).
- **MSE** – Mean Squared Error.
- **RMSE** – Root Mean Squared Error (original target units).
- **R²** – Coefficient of determination.
- **Adjusted R²** – Penalized R² accounting for the number of features after preprocessing.

All reported test-set metrics are computed on the held-out 20% test split. Cross-validation metrics are computed on training folds only.

---

## 7. Results Summary

> **Note:** Detailed results are available in the per-dataset CSV files and the full technical report (`23MID0043_Lab01_Report.pdf`).

### Ames Housing
- **Target:** `SalePrice` (USD)
- **Rows:** 1,460 total; 1,168 train / 292 test
- **Best model:** Gradient Boosting Regressor
- **Test MAE:** $16,555.67 | **Test RMSE:** $26,503.77 | **Test R²:** 0.908
- **CV RMSE:** $27,708.84 | **CV R²:** 0.867
- **Key predictors:** Above-ground living area (`GrLivArea`), overall quality (`OverallQual`), neighborhood, garage capacity.

### California Housing
- **Target:** `MedHouseVal` (median district house value in USD 100,000s)
- **Rows:** 20,640 total; 16,512 train / 4,128 test
- **Best model:** Random Forest Regressor
- **Test MAE:** 0.327 ($32,700) | **Test RMSE:** 0.505 ($50,500) | **Test R²:** 0.805
- **CV RMSE:** 0.511 | **CV R²:** 0.805
- **Key predictors:** Median income, house age, average rooms, latitude/longitude.

### UCI Real Estate
- **Target:** `Y house price of unit area` (10,000 NTD per ping)
- **Rows:** 414 total; 331 train / 83 test
- **Best model:** Random Forest Regressor
- **Test MAE:** 3.913 | **Test RMSE:** 5.669 | **Test R²:** 0.808
- **CV RMSE:** 7.830 | **CV R²:** 0.664
- **Key predictors:** Distance to nearest MRT station, number of convenience stores, latitude/longitude, house age.

---

## 8. Reproducibility Notes

- All random operations use `random_state=42`.
- Preprocessing is fit only on the training split.
- The test set is used exactly once for final model comparison.
- Model selection uses cross-validated training performance, not test performance.
- Raw data files (`train.csv`) are kept unchanged.

---

## 9. References

- De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. *Journal of Statistics Education*, 19(3).
- scikit-learn documentation. California Housing dataset.
- UCI Machine Learning Repository. Real Estate Valuation dataset (ID 477).
- James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. *An Introduction to Statistical Learning with Applications in Python*.
