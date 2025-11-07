# Model-based Machine Learning Homework Submission


## Student Info
- Name: Xinliu Zhong
- Contact: xinliu.zhong@emory.edu

## Selected Question
- HW 3: Model-based Bias Removal in Machine Learning using Synthetic Blood Pressure Data

## Key Insights
- Quadratic SBP/DBP fits match the tabulated means with MSE ≤ 0.90 mmHg² (SBP R²≈0.999, DBP R²≈0.926), while the Gaussian DBP model captures the mid-life peak with the lowest residual (MSE≈0.80 mmHg²).
- Sigmoidal SBP parameters (`Smax`≈149 mmHg, `k`≈0.016 yr⁻¹) imply an inflection in early adulthood, highlighting that additional pediatric data would better constrain the half-maximum age.
- Synthetic SBP/DBP sampling via sex-specific bivariate normals reproduces the desired prevalence scenarios and enables controlled bias analyses.
- Without mitigation, female recall collapses (<4% at a 70/30 split) despite seemingly strong accuracy/AUC, revealing a severe fairness gap.
- Class-weighted logistic regression restores minority recall (≈0.58 for both sexes) with negligible loss in AUC, illustrating a simple model-based bias countermeasure.

## Comparative Model Performance
### Blood Pressure vs. Age Fits

| Model | MSE (mmHg^2) | R^2 |
| --- | --- | --- |
| SBP (Quadratic) | 0.065 | 0.999 |
| SBP (Sigmoid) | 0.065 | 0.999 |
| DBP (Quadratic) | 0.898 | 0.926 |
| DBP (Gaussian) | 0.801 | 0.934 |

### Sex Classification from Synthetic BP
| Male Ratio | Method | Accuracy | F1 | AUC | Recall (Male) | Recall (Female) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | Unweighted | 0.585 | 0.586 | 0.622 | 0.587 | 0.582 |
| 0.7 | Unweighted | 0.702 | 0.822 | 0.613 | 0.987 | 0.037 |
| 0.9 | Unweighted | 0.900 | 0.947 | 0.625 | 1.000 | 0.000 |
| 0.5 | Class-weighted | 0.585 | 0.586 | 0.622 | 0.587 | 0.582 |
| 0.7 | Class-weighted | 0.584 | 0.663 | 0.613 | 0.586 | 0.579 |
| 0.9 | Class-weighted | 0.584 | 0.716 | 0.625 | 0.583 | 0.596 |

## Relevance to Model-based Machine Learning
- The notebook encodes physiological priors (quadratic/sigmoid/Gaussian BP curves) directly from the literature, providing interpretable parameters that align with age-driven dynamics.
- Synthetic data generation uses explicit probabilistic models (bivariate normals with prescribed covariances) to study how prevalence shifts affect downstream classifiers.
- Bias diagnostics (ROC curves, recall gaps) demonstrate how model-based reasoning guides fairness interventions by exposing mismatches between data-generating assumptions and classifier behavior.

## Suggestions for Future Modeling Improvements
- Incorporate age- and treatment-specific covariates (BMI, antihypertensive use) to extend the generative model beyond two vital signs.
- Replace the static correlation assumption with copula-based sampling to encode non-linear SBP/DBP dependence.
- Explore Bayesian hierarchical models that pool sex-specific parameters while allowing subgroup variations, then propagate posterior uncertainty through the classifier.
- Evaluate threshold-tuning or cost-sensitive decision rules that explicitly optimize balanced accuracy under deployment prevalence.

## Reproduction
1. (Optional) Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`.
2. Install dependencies: `pip install numpy pandas matplotlib scipy scikit-learn jupyter`.
3. Run the notebook end-to-end.

