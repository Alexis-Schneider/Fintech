# Credit Card Customer Churn Prediction

**FinTech — Final Assignment, Politecnico di Milano**


---

## 1. Problem

A bank is losing credit-card customers and wants to identify likely churners **early enough to act** — before the customer has effectively left. This is a binary classification task with two complications:

1. **Class imbalance** — only 16.1% of customers churn (1,627 / 10,127).
2. **A business asymmetry** — a missed churner (false negative) costs far more than an unnecessary retention call (false positive).

The headline tension this project resolves: the most predictive features in the dataset are **lagging indicators** (transaction volume collapsing), which fire too late to be actionable. So the project trains every model **twice** — once with these features (best-case predictor) and once without (deployable early-warning system) — and quantifies the gap.

---

## 2. Key results (one-glance)

| Configuration | Model | AP | ROC-AUC | Recall* | Precision* |
|---|---|---:|---:|---:|---:|
| FULL (with transactional) | Random Forest | **0.9403** | 0.9860 | 0.76 | 0.93 |
| FULL | MLP | 0.8864 | 0.9707 | 0.88 | 0.70 |
| PROACTIVE (no transactional) | Random Forest | 0.5625 | 0.8446 | 0.62 | 0.51 |
| PROACTIVE | MLP | 0.5671 | 0.8345 | 0.74 | 0.39 |

\* at the default 0.50 threshold.

- The **~0.38 AP drop** between FULL and PROACTIVE is the structural "price of being proactive."
- On PROACTIVE, **RF and MLP are statistically indistinguishable** (95% bootstrap CIs overlap almost entirely). The *configuration* choice matters about an order of magnitude more than the *model-family* choice.
- **Feature selection (21 → 10)** slightly *improves* both models (RF 0.5625 → 0.5811, MLP 0.5671 → 0.5838).
- **Recommended production model:** calibrated reduced RF (`RF_PRO_Reduced_Calibrated`), with the operating threshold read off the cost-sensitivity table rather than fixed at the F2 optimum.

---

## 3. Dataset

`Dataset1.xlsx` — 10,127 rows, ~20 raw columns after cleaning.

**Cleaning:** two `Naive_Bayes_Classifier_*` columns are dropped — they encode the target via a prior model's output and are a textbook **leakage trap** in this dataset. `CLIENTNUM` (identifier) is also dropped. The target is `Attrition_Flag` → `Churn` (1 = attrited, 0 = existing).

**Two feature configurations:**

| Config | Transactional features | Feature count (after engineering) | Use case |
|---|---|---:|---|
| `FULL` | included | 25 | best-case predictor / benchmark |
| `PROACTIVE` | excluded | 21 | deployable early-warning system |

The four excluded "lagging" features are `Total_Trans_Amt`, `Total_Trans_Ct`, `Total_Ct_Chng_Q4_Q1`, `Total_Amt_Chng_Q4_Q1`.

**Engineered features** (present in both configs): `Balance_to_Limit_Ratio`, `Contacts_per_Inactive_Month`. Ordinal encoding for `Education_Level` / `Income_Category`; one-hot for `Marital_Status` / `Card_Category`.

---

## 4. Methodology

1. **Dual analysis** — every model trained on both `FULL` and `PROACTIVE`.
2. **Imbalance-aware metrics** — **Average Precision (AP)** as the primary selection metric, ROC-AUC for ranking quality, **F2-score** at deployment (recall weighted 4× over precision).
3. **No data leakage** — stratified splits, scaler fit on train only, all decision thresholds tuned on out-of-fold / inner-validation predictions, never on the test set.
4. **Two ML families** — Random Forest (GridSearchCV on AP) and a regularized MLP (BatchNorm → ReLU → Dropout → L2, EarlyStopping, ReduceLROnPlateau).
5. **Feature selection** — RFECV with a **1-sigma parsimony rule** (smallest subset within one std of the CV peak), cross-checked with SHAP (RF) and permutation importance (MLP).
6. **Calibration + cost-sensitive thresholding** — isotonic calibration of the RF, then an expected-cost analysis across FN:FP ratios.

### Why AP and not accuracy
At 16% positives, a model predicting "no churn" for everyone scores 84% accuracy and is useless. AP (area under the precision-recall curve) reflects performance on the minority class; the random baseline equals the prior (0.16), so the tuned RF's PROACTIVE AP of 0.56 is a ~3.5× lift over chance.

### Overfitting diagnosis (the OOB trick)
The tuned PROACTIVE RF shows an alarming in-sample→test AP gap of +0.33 — but this is a **bootstrap artefact**: each sample is seen by ~63% of the trees, inflating the in-sample score. The honest **out-of-bag** estimate (each sample scored only by trees that never saw it) gives OOB AP = 0.577 vs test 0.563, a true generalization gap of just **+0.014**. The model generalizes cleanly.

---

## 5. Feature selection

RFECV on the PROACTIVE config peaks at **k = 15** (CV AP = 0.574). The 1-sigma rule selects the smallest subset within one std of that peak: **k = 10**.

Selected features:
`Customer_Age`, `Months_on_book`, `Total_Relationship_Count`, `Months_Inactive_12_mon`, `Contacts_Count_12_mon`, `Credit_Limit`, `Total_Revolving_Bal`, `Avg_Open_To_Buy`, `Avg_Utilization_Ratio`, `Balance_to_Limit_Ratio`.

The reduced models match or slightly beat the full-PROACTIVE models with half the inputs:

| Model | AP (21 feat) | AP (10 feat) |
|---|---:|---:|
| RF PROACTIVE | 0.5625 | **0.5811** |
| MLP PROACTIVE | 0.5671 | **0.5838** |

**Top churn drivers** (consistent across SHAP and MLP permutation importance): `Total_Revolving_Bal` (low balance → disengaging), `Contacts_Count_12_mon` (frequent service contacts → dissatisfaction), `Total_Relationship_Count` (single-product customers leave more easily).

---

## 6. Calibration and business thresholding

Random Forests push probabilities toward 0 and 1, so the raw score is a poor probability. **Isotonic calibration** improves the Brier score from 0.1140 to 0.0971 while preserving ranking (AUC 0.846 → 0.843), making the threshold interpretable as a real churn probability.

**Cost-sensitivity of the optimal threshold** (calibrated RF, PROACTIVE):

| FN:FP cost ratio | τ* | Recall | Precision |
|---|---:|---:|---:|
| 1:1 | 0.462 | 0.39 | 0.67 |
| 1:5 | 0.188 | 0.75 | 0.40 |
| 1:10 | 0.110 | 0.85 | 0.33 |
| 1:20 | 0.050 | 0.92 | 0.25 |

The F2-optimal threshold (τ ≈ 0.098) catches **280 of 325** test churners (recall 0.86) but fires 884 alerts — the recall-aggressive extreme. The production operating point should be picked from this table given the bank's real cost ratio.

---

## 7. Repository structure

```
.
├── Fintech_projet.ipynb      # main notebook (EDA → models → selection → calibration → cost analysis)
├── Dataset1.xlsx             # BankChurners dataset (not redistributed if license-restricted)
├── README.md                 # this file
└── presentation.pptx         # slide deck framing problem, approach, results, insights
```

The notebook is organized in numbered sections:

| Section | Content |
|---|---|
| 1 | Exploratory data analysis (imbalance, distributions, correlations) |
| 2 | Dual preprocessing pipeline + class-weight rationale |
| 3 | Random Forest (baselines, GridSearchCV, OOB diagnosis) |
| 4 | Multi-Layer Perceptron (architecture, hyperparameter search) |
| 5 | Model comparison (RF vs MLP × FULL vs PROACTIVE) + bootstrap CIs |
| 6 | Feature selection (RFECV + 1-sigma rule + SHAP) |
| 7 | Impact of feature selection + probability calibration |
| 8 | Business threshold optimization (F2) + expected-cost analysis + caveats |
| 9 | Conclusions and recommended deployment |

---

## 8. How to run

**Requirements** (Python 3.10):

```bash
pip install numpy pandas matplotlib seaborn scikit-learn shap tensorflow openpyxl
```

Then place `Dataset1.xlsx` next to the notebook and run all cells top to bottom:

```bash
jupyter notebook Fintech_projet.ipynb
```

Runtime is a few minutes on CPU. A fixed seed (`SEED = 42`) is set for NumPy and TensorFlow; the Random Forest results are deterministic, while the MLP carries minor run-to-run variance (the figures in Section 9 reflect the committed run).

---

## 9. Caveats and limitations

- **Collinearity in the retained set** — `Credit_Limit ≈ Total_Revolving_Bal + Avg_Open_To_Buy` by construction, and the two ratio features are further functions of the same quantities. Harmless for RF, but RFECV optimizes AP, not orthogonality.
- **F2 threshold is the recall-aggressive extreme** — at τ ≈ 0.10 the model alerts on ~44% of the customer base; the deployment threshold should come from the cost table.
- **Asymmetric reduced-model protocol** — the reduced RF is re-tuned; the reduced MLP reuses the best PROACTIVE hyperparameters (re-running a Keras search per subset is expensive).
- **`Unknown` encoded as the lowest ordinal level** for `Education_Level` / `Income_Category` — it is missingness, not a low rank.
- **Single stratified 80/20 split** — repeated splits or nested CV would tighten the confidence intervals.

---

## 10. Author

Alexis — MSc Engineering, exchange semester at Politecnico di Milano.
Course: FinTech (Prof. Marazzina).
