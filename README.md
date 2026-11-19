# Predicting first-session ESWL outcomes in ureteral stones

R analysis code and a Shiny research application for the study *Interpretable machine learning model using imaging and clinical features to predict first-session extracorporeal shock wave lithotripsy (ESWL) outcomes in ureteral stones: a multicenter retrospective study*.

The Jinzhou development cohort comprised 829 patients, with 621 assigned to training and 208 to a held-out internal test set. The Huludao external-validation cohort comprised 147 patients. The model estimates the probability of composite treatment success after the first ESWL session. [Open the research application](https://eswl.shinyapps.io/ESWL-Prediction/)

## Outcomes and predictors

The primary outcome required all six criteria: residual fragments smaller than 4 mm on imaging at 3–7 days, no repeat ESWL within four weeks, temperature below 38 °C, absence of steinstrasse, a 24-hour pain score of 3 or less, and macroscopic hematuria beginning to improve within 48 hours without subsequent worsening. The stone-control sensitivity analysis used the first two criteria and retained the original model and threshold.

The six pretreatment predictors were stone volume (mm³), mean stone attenuation (HU), ureteral wall thickness (mm), anteroposterior renal pelvic diameter (mm), subcutaneous fat thickness (mm), and ureteral stone location (distal, mid, or proximal). The positive class is success. Classification uses a fixed threshold of **0.7307063937187195** (0.731 in the manuscript).

## Run an analysis

The manuscript reports R 4.4.2. Run scripts from the repository root in a fresh R session. The patient-level inputs and the original fitted workflows are held separately. Use the original package environment when loading a serialized model, particularly for xgboost, parsnip, recipes and workflows. The CV script writes the session information from each run.

The following commands install the principal packages for development, evaluation, reporting and the app. Install only the group needed for a given analysis if preferred. Use matching versions from the study environment when reproducing an existing fit.

```r
install.packages(c(
  "tidymodels", "bonsai", "Boruta", "randomForest", "ranger",
  "rpart", "glmnet", "kernlab", "kknn", "nnet", "xgboost", "lightgbm",
  "rms", "segmented", "forestploter", "pROC", "patchwork", "cowplot",
  "fastshap", "shapviz", "ggbeeswarm", "officer", "flextable", "shiny"
))
```

| Task | Inputs to place locally |
|---|---|
| Fixed development split, Boruta, forest plot, splines and breakpoints | Original development CSV at `data/831ESWL.csv`. The filename is historical; the manuscript reports 829 included patients. |
| CV, baseline table and SHAP | The split script writes `01_traindata.rds` and `01_testdata_SEALED.rds` under `output/feature_selection/`. The Figure 4 script also reads the saved Boruta object there. |
| Primary three-cohort evaluation | `data/复合碎石成功的三个队列.RData` containing `training_data`, `test_data` and `external_data`. |
| Stone-control evaluation | `data/单纯碎石成功的三队列.RDATA` containing `training_data_sfr`, `test_data_sfr` and `external_data_sfr`. Table S3 needs both outcome workspaces. |
| Locked-model evaluation and interpretation | `04_XGBoost_best_CV_refit_621.rds` and `00_ESWL_all_9_best_CV_refit_models.rds` under `output/model_benchmark_9_CV_FORMAL/13_final_models/`. Tables 3 and S4 also read `04_development_OOF_Youden_thresholds.csv` from the parent directory. |

To reproduce the development comparison with the original development CSV:

```sh
Rscript analysis/01_split_and_boruta.R
Rscript analysis/02_nine_model_cross_validation.R
Rscript figures/figure06_model_comparison.R
```

The first step creates the fixed split and screens pretreatment variables in the training cohort. The second step compares nine algorithms using the same stratified five-fold cross-validation repeated ten times. It averages each patient's out-of-fold (OOF) predictions across repeats, derives the Youden threshold and saves the CV results. These are post-selection development estimates. The manuscript describes manual selection of XGBoost after review of the training results. The original final-refit script and patient-level outcome-construction scripts are outside this source release. Downstream analyses load the previously fitted workflows.

To reproduce the primary validation figures when the corresponding inputs are available:

```sh
Rscript figures/figure07_roc_and_confusion.R
Rscript figures/figure08_calibration_and_dca.R
```

## Manuscript and supplementary outputs

| Output | Script | Analysis |
|---|---|---|
| Figure 3 | [Univariable forest plot](figures/figure03_univariable_forest.R) | Univariable logistic models for 19 factors, with overall likelihood-ratio tests. |
| Figure 4 | [Split and Boruta](analysis/01_split_and_boruta.R) · [Assemble Figure 4](figures/figure04_boruta_and_associations.R) | Training-set importance and pairwise associations among confirmed predictors. |
| Figure 5 | [Adjusted splines](figures/figure05_adjusted_splines.R) | Exploratory associations for the five continuous predictors in the development cohort. |
| Figure 6 | [Nine-model CV](analysis/02_nine_model_cross_validation.R) · [Model comparison](figures/figure06_model_comparison.R) | Fold-level AUC and patient-level OOF Brier score, calibration intercept and calibration slope. Model order uses unrounded CV mean AUC. |
| Figure 7 | [ROC and confusion matrices](figures/figure07_roc_and_confusion.R) | Apparent training, internal test and external performance at the fixed threshold. |
| Figure 8 | [Calibration and decision curves](figures/figure08_calibration_and_dca.R) | Logistic calibration curves, grouped observed proportions and decision curves from 0.10 to 0.90. |
| Figure 9 | [Global and individual SHAP](figures/figure09_shap.R) | Probability-scale FastSHAP in the training cohort and representative patients at the median, 10th and 90th percentiles of predicted probability. |
| Table 1 | [Baseline characteristics](tables/table01_cohort_characteristics.R) | Development split, descriptive summaries and absolute standardized mean differences (SMDs). |
| Table 2 | [Breakpoint analysis](tables/table02_breakpoint_analysis.R) | Adjusted two-piecewise models for mean CT attenuation and subcutaneous fat thickness. |
| Table 3 | [Locked-model performance](tables/table03_model_performance.R) | Discrimination, classification, calibration and Brier score. Exports include an AUC row and additional diagnostic plots. |
| Figure S1 | [Stone-control outcome](supplement/figureS1_stone_control.R) | ROC and confusion matrices using the alternative reference outcome. |
| Figure S2 | [SHAP stability](supplement/figureS2_shap_stability.R) | Five-fold analysis and assessment SHAP patterns for each of the six predictors. |
| Table S3 | [Three-cohort characteristics](supplement/tableS3_three_cohort_characteristics.R) | Six predictors, two outcomes and SMDs. |
| Table S4 | [Stone-control performance](supplement/tableS4_stone_control_performance.R) | Fixed-model evaluation with stone control as the reference outcome. |

Figures 1 and 2 are the study diagram and CT measurement illustration. Table S1 describes center protocols. Table S2 reports overlapping component frequencies from clinical follow-up. The source materials for these four items were not supplied as analysis scripts.

## Interpretation and version checks

- **Calibration.** In Figure 8, the shaded pointwise confidence bands use the logistic model's link-scale standard error, transformed back to probability. Grouped observed proportions use exact binomial intervals. Table 3 uses 2,000 patient-level bootstrap draws for calibration-in-the-large, slope and Brier intervals.
- **Case mix.** The Table S3 script computes a single multicategory SMD for stone location from pooled multinomial covariance. The supplementary counts give 0.130 for training versus internal test and 0.199 for training versus external validation. The supplied Word version lists 0.120 and 0.217.
- **Model identity.** The expected MD5 of the standalone workflow is `25e87ace906a55706bda38e6ec29e8b2`. The nine-model bundle has MD5 `778b32d90bb90ff37d3d849a6a27bae4`. Its saved XGBoost engine matches the engine in the standalone workflow. Figure 7's displayed AUCs are 0.897, 0.863 and 0.980 for the training, internal test and external cohorts.
- **Code release.** The scripts use project-relative paths. Figure labels and output filenames have been aligned for publication. The source files and archived model identifiers were checked during preparation; the clinical analyses were not rerun for this release.

## Shiny research application

The app uses the standalone fitted workflow, reports the predicted probability and fixed-threshold classification, and groups XGBoost gain by clinical predictor. The app's individual tree contributions are on the log-odds scale. Figure 9 uses probability-scale Monte Carlo SHAP.

With the standalone workflow in the local model directory described above, run the following in R from the repository root:

```r
model_dir <- file.path("output", "model_benchmark_9_CV_FORMAL", "13_final_models")
model_file <- file.path(model_dir, "04_XGBoost_best_CV_refit_621.rds")
Sys.setenv(ESWL_MODEL_PATH = normalizePath(model_file, mustWork = TRUE))
shiny::runApp("app")
```

The [application source](app/app.R) also accepts the model beside `app.R` when started from that directory. The app was developed for research use.
