# ==============================================================================
# ESWL: comparison of 9 candidate algorithms in the development cohort only
# ==============================================================================
# Outcome: success = 1 (positive event); .pred_1 = probability of success.
#
# Fixed analysis plan
#   1. Read the already saved 621-patient development cohort directly.
#   2. Never recreate the 75:25 split in this script.
#   3. The six predictors were fixed by conventional Boruta in those 621 cases:
#      Stone volume, mean CT attenuation, ureteral wall thickness,
#      continuous hydronephrosis measure, subcutaneous fat thickness,
#      and stone location.
#   4. Compare all algorithms with the same repeated stratified 5-fold CV.
#   5. Do not extract, predict, evaluate, rank by, or save the 208-patient test set.
#
# Interpretation
#   Because Boruta used all 621 development cases before CV, these are
#   post-selection CV results for comparing algorithms. Final unbiased model
#   performance must be reported later in the sealed internal test cohort,
#   after one algorithm and its tuning rule have been locked.
# ==============================================================================


# ==============================================================================
# 0. User settings
# ==============================================================================

QUICK_MODE <- FALSE  # TRUE: code check; FALSE: formal 5-fold CV x 10 repeats

training_file <- paste0(
  "output/feature_selection/",
  "01_traindata.rds"
)

output_dir <- if (QUICK_MODE) {
  "output/model_benchmark_9_CV_QUICK"
} else {
  "output/model_benchmark_9_CV_FORMAL"
}

SEED_CV <- 20260902L
CV_FOLDS <- 5L
CV_REPEATS <- if (QUICK_MODE) 1L else 10L

EXPECTED_DEVELOPMENT_N <- 621L
EXPECTED_TEST_N <- 208L
EXPECTED_SUCCESS_N <- 520L
EXPECTED_FAILURE_N <- 101L

# Models within 0.01 of the largest mean CV AUC are flagged as near-best.
# This is a review aid, not an automatic selection rule.
AUC_NEAR_BEST_MARGIN <- 0.01

grid_size <- if (QUICK_MODE) {
  c(DT = 5L, RF = 5L, XGBoost = 5L, RSVM = 5L,
    MLP = 5L, LightGBM = 5L, ENet_penalty = 3L,
    ENet_mixture = 3L, kNN = 5L)
} else {
  c(DT = 20L, RF = 20L, XGBoost = 30L, RSVM = 25L,
    MLP = 20L, LightGBM = 30L, ENet_penalty = 10L,
    ENet_mixture = 5L, kNN = 12L)
}


# ==============================================================================
# 1. Packages
# ==============================================================================

required_packages <- c(
  "tidymodels", "readr", "bonsai", "rpart", "randomForest", "xgboost",
  "kernlab", "nnet", "lightgbm", "glmnet", "kknn"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0L) {
  stop("Install missing packages first: ", paste(missing_packages, collapse = ", "))
}

suppressPackageStartupMessages({
  library(tidymodels)
  library(bonsai)  # registers the LightGBM engine
})

if (!dir.exists(output_dir) &&
    !dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)) {
  stop("Cannot create output directory: ", output_dir)
}


# ==============================================================================
# 2. Read and validate data
# ==============================================================================

message("1/8  Reading and checking the saved development cohort ...")

if (!file.exists(training_file)) {
  stop("Training file does not exist: ", training_file)
}

train_raw <- readRDS(training_file)
training_file_md5 <- unname(tools::md5sum(training_file))

if (!is.data.frame(train_raw)) {
  stop("01_traindata.rds must contain one data frame/tibble.")
}

required_columns <- c(
  "success",
  "Stone volume",
  "Mean CT attenuation",
  "Ureteral wall thickness",
  "Degree of hydronephrosis",
  "Subcutaneous fat thickness",
  "Stone location"
)

missing_columns <- setdiff(required_columns, names(train_raw))
if (length(missing_columns) > 0L) {
  stop("Missing required columns: ", paste(missing_columns, collapse = ", "))
}

if (nrow(train_raw) != EXPECTED_DEVELOPMENT_N) {
  stop(
    "Expected ", EXPECTED_DEVELOPMENT_N,
    " development cases, but found ", nrow(train_raw), "."
  )
}

parse_outcome <- function(x) {
  raw <- trimws(tolower(as.character(x)))
  out <- dplyr::case_when(
    raw %in% c("yes", "1", "success", "successful") ~ "1",
    raw %in% c("no", "0", "failure", "failed", "unsuccessful") ~ "0",
    TRUE ~ NA_character_
  )

  if (anyNA(out)) {
    bad <- unique(raw[is.na(out)])
    stop(
      "Stone success contains missing/unrecognized values: ",
      paste(utils::head(bad, 10L), collapse = ", ")
    )
  }
  out
}

strict_numeric <- function(x, name) {
  raw <- trimws(as.character(x))
  raw[is.na(x) | raw == ""] <- NA_character_
  out <- suppressWarnings(as.numeric(raw))
  bad <- !is.na(raw) & (is.na(out) | !is.finite(out))

  if (any(bad)) {
    stop(
      name, " contains non-numeric/non-finite values: ",
      paste(utils::head(unique(raw[bad]), 10L), collapse = ", ")
    )
  }
  out
}

outcome_01 <- parse_outcome(train_raw[["success"]])


# ==============================================================================
# 3. Select the six fixed predictors
# ==============================================================================

development_ids <- seq_len(nrow(train_raw))

train_data <- tibble::tibble(
  Case_ID = development_ids,
  success = factor(outcome_01, levels = c("1", "0")),
  Stone_volume = strict_numeric(train_raw[["Stone volume"]], "Stone volume"),
  Mean_CT_attenuation = strict_numeric(
    train_raw[["Mean CT attenuation"]], "Mean CT attenuation"
  ),
  Ureteral_wall_thickness = strict_numeric(
    train_raw[["Ureteral wall thickness"]], "Ureteral wall thickness"
  ),
  Hydronephrosis_measure = strict_numeric(
    train_raw[["Degree of hydronephrosis"]], "Degree of hydronephrosis"
  ),
  Subcutaneous_fat_thickness = strict_numeric(
    train_raw[["Subcutaneous fat thickness"]], "Subcutaneous fat thickness"
  ),
  Stone_location = factor(
    dplyr::na_if(trimws(as.character(train_raw[["Stone location"]])), "")
  )
)

stopifnot(
  nrow(train_data) == EXPECTED_DEVELOPMENT_N,
  identical(levels(train_data$success), c("1", "0")),
  dplyr::n_distinct(train_data$success) == 2L,
  !anyDuplicated(train_data$Case_ID)
)

outcome_counts <- table(train_data$success)
if (unname(outcome_counts[["1"]]) != EXPECTED_SUCCESS_N ||
    unname(outcome_counts[["0"]]) != EXPECTED_FAILURE_N) {
  stop(
    "Expected success/failure counts of 520/101, but found ",
    outcome_counts[["1"]], "/", outcome_counts[["0"]], "."
  )
}

if (anyNA(train_data)) {
  stop("The saved development data contain missing values in the outcome/predictors.")
}

if (min(table(train_data$success)) < CV_FOLDS) {
  stop("At least one outcome class has fewer cases than the number of CV folds.")
}

missingness <- train_data |>
  dplyr::summarise(dplyr::across(dplyr::everything(), ~ sum(is.na(.x)))) |>
  tidyr::pivot_longer(
    dplyr::everything(), names_to = "Variable", values_to = "Missing_n"
  )

message(
  "Development n = ", nrow(train_data),
  "; test n = ", EXPECTED_TEST_N, " (sealed file not read)"
)
print(table(train_data$success))


# ==============================================================================
# 4. Recipes and shared repeated CV
# ==============================================================================

message("2/8  Creating recipes and shared CV folds ...")

make_recipe <- function(scale_predictors = FALSE) {
  rec <- recipes::recipe(success ~ ., data = train_data) |>
    recipes::update_role(Case_ID, new_role = "case_id") |>
    recipes::step_novel(
      recipes::all_nominal_predictors(), new_level = "__NEW__"
    ) |>
    recipes::step_dummy(recipes::all_nominal_predictors()) |>
    recipes::step_zv(recipes::all_predictors())

  # Scale the complete design matrix, including dummy variables. This is used
  # only for distance-/coefficient-based methods.
  if (scale_predictors) {
    rec <- rec |> recipes::step_normalize(recipes::all_numeric_predictors())
  }
  rec
}

recipe_basic <- make_recipe(FALSE)
recipe_scaled <- make_recipe(TRUE)

set.seed(SEED_CV)
cv_folds <- rsample::vfold_cv(
  train_data,
  v = CV_FOLDS,
  repeats = CV_REPEATS,
  strata = success
)

expected_resamples <- CV_FOLDS * CV_REPEATS
stopifnot(nrow(cv_folds) == expected_resamples)

count_processed_predictors <- function(data) {
  prepared <- recipes::prep(
    recipe_basic, training = data, retain = TRUE, verbose = FALSE
  )
  ncol(
    recipes::bake(
      prepared,
      new_data = NULL,
      recipes::all_predictors(),
      composition = "data.frame"
    )
  )
}

# step_zv() can remove a rare location dummy within a fold. Use an mtry upper
# bound that is valid in every analysis fold.
mtry_upper <- min(purrr::map_int(
  cv_folds$splits,
  ~ count_processed_predictors(rsample::analysis(.x))
))

if (!is.finite(mtry_upper) || mtry_upper < 1L) {
  stop("No fold-safe mtry value is available.")
}

message("Fold-safe number of processed predictors = ", mtry_upper)


# ==============================================================================
# 5. Model specifications and grids
# ==============================================================================

message("3/8  Defining models and tuning grids ...")

make_workflow <- function(recipe, model) {
  workflows::workflow() |>
    workflows::add_recipe(recipe) |>
    workflows::add_model(model)
}

latin_grid <- function(parameters, size, seed) {
  set.seed(seed)
  dials::grid_latin_hypercube(parameters, size = size)
}

spec_logistic <- parsnip::logistic_reg(mode = "classification") |>
  parsnip::set_engine("glm")

spec_dt <- parsnip::decision_tree(
  mode = "classification",
  tree_depth = tune(), min_n = tune(), cost_complexity = tune()
) |>
  parsnip::set_engine("rpart")

spec_rf <- parsnip::rand_forest(
  mode = "classification",
  mtry = tune(), trees = 1000L, min_n = tune()
) |>
  parsnip::set_engine("randomForest", importance = TRUE)

spec_xgb <- parsnip::boost_tree(
  mode = "classification",
  mtry = tune(), trees = 1000L, min_n = tune(), tree_depth = tune(),
  learn_rate = tune(), loss_reduction = tune(), sample_size = tune(),
  stop_iter = 25L
) |>
  parsnip::set_engine("xgboost", validation = 0.20, nthread = 1L)

spec_rsvm <- parsnip::svm_rbf(
  mode = "classification", cost = tune(), rbf_sigma = tune()
) |>
  parsnip::set_engine("kernlab")

spec_mlp <- parsnip::mlp(
  mode = "classification",
  hidden_units = tune(), penalty = tune(), epochs = tune()
) |>
  parsnip::set_engine("nnet", MaxNWts = 5000L, trace = FALSE)

spec_lgbm <- parsnip::boost_tree(
  mode = "classification",
  tree_depth = tune(), trees = tune(), learn_rate = tune(), mtry = tune(),
  min_n = tune(), loss_reduction = tune()
) |>
  parsnip::set_engine("lightgbm", num_threads = 1L)

spec_enet <- parsnip::logistic_reg(
  mode = "classification", penalty = tune(), mixture = tune()
) |>
  parsnip::set_engine("glmnet", standardize = FALSE)

spec_knn <- parsnip::nearest_neighbor(
  mode = "classification",
  neighbors = tune(), weight_func = "rectangular", dist_power = 2
) |>
  parsnip::set_engine("kknn")

workflows <- list(
  Logistic = make_workflow(recipe_basic, spec_logistic),
  DT = make_workflow(recipe_basic, spec_dt),
  RF = make_workflow(recipe_basic, spec_rf),
  XGBoost = make_workflow(recipe_basic, spec_xgb),
  RSVM = make_workflow(recipe_scaled, spec_rsvm),
  MLP = make_workflow(recipe_scaled, spec_mlp),
  LightGBM = make_workflow(recipe_basic, spec_lgbm),
  ENet = make_workflow(recipe_scaled, spec_enet),
  kNN = make_workflow(recipe_scaled, spec_knn)
)

grids <- list(
  Logistic = NULL,
  DT = latin_grid(
    dials::parameters(
      dials::tree_depth(range = c(2L, 6L)),
      dials::min_n(range = c(5L, 30L)),
      dials::cost_complexity(range = c(-6, -1))
    ), grid_size[["DT"]], 2002L
  ),
  RF = latin_grid(
    dials::parameters(
      dials::mtry(range = c(1L, mtry_upper)),
      dials::min_n(range = c(5L, 30L))
    ), grid_size[["RF"]], 2003L
  ),
  XGBoost = latin_grid(
    dials::parameters(
      dials::mtry(range = c(1L, mtry_upper)),
      dials::min_n(range = c(5L, 30L)),
      dials::tree_depth(range = c(1L, 4L)),
      dials::learn_rate(range = c(-3, -1)),
      dials::loss_reduction(range = c(-4, 0)),
      dials::sample_prop(range = c(0.70, 1.00))
    ), grid_size[["XGBoost"]], 2004L
  ),
  RSVM = latin_grid(
    dials::parameters(
      dials::cost(range = c(-5, 5)),
      dials::rbf_sigma(range = c(-5, 0))
    ), grid_size[["RSVM"]], 2005L
  ),
  MLP = latin_grid(
    dials::parameters(
      dials::hidden_units(range = c(2L, 12L)),
      dials::penalty(range = c(-5, -1)),
      dials::epochs(range = c(50L, 200L))
    ), grid_size[["MLP"]], 2006L
  ),
  LightGBM = latin_grid(
    dials::parameters(
      dials::tree_depth(range = c(1L, 4L)),
      dials::trees(range = c(100L, 800L)),
      dials::learn_rate(range = c(-3, -1)),
      dials::mtry(range = c(1L, mtry_upper)),
      dials::min_n(range = c(5L, 30L)),
      dials::loss_reduction(range = c(-4, 0))
    ), grid_size[["LightGBM"]], 2007L
  ),
  ENet = dials::grid_regular(
    dials::parameters(
      dials::penalty(range = c(-5, 0)),
      dials::mixture(range = c(0, 1))
    ),
    levels = c(
      penalty = grid_size[["ENet_penalty"]],
      mixture = grid_size[["ENet_mixture"]]
    )
  ),
  kNN = dials::grid_regular(
    dials::parameters(dials::neighbors(range = c(3L, 25L))),
    levels = grid_size[["kNN"]]
  )
)


# ==============================================================================
# 6. Fit all models with the same resamples
# ==============================================================================

message("4/8  Fitting nine models ...")

cv_metrics <- yardstick::metric_set(yardstick::roc_auc)
cv_control <- tune::control_grid(
  save_pred = TRUE,
  save_workflow = FALSE,
  verbose = TRUE
)

fit_one_model <- function(model_name, seed) {
  message("  ", model_name, " ...")
  set.seed(seed)

  if (is.null(grids[[model_name]])) {
    tune::fit_resamples(
      workflows[[model_name]],
      resamples = cv_folds,
      metrics = cv_metrics,
      control = cv_control
    )
  } else {
    tune::tune_grid(
      workflows[[model_name]],
      resamples = cv_folds,
      grid = grids[[model_name]],
      metrics = cv_metrics,
      control = cv_control
    )
  }
}

model_order <- names(workflows)
results <- purrr::map2(
  model_order,
  3001L + seq_along(model_order) - 1L,
  fit_one_model
)
names(results) <- model_order


# ==============================================================================
# 7. Extract and strictly validate the selected configuration for each model
# ==============================================================================

message("5/8  Extracting CV results ...")

get_best_config <- function(result) {
  best <- tune::collect_metrics(result) |>
    dplyr::filter(.metric == "roc_auc", is.finite(mean)) |>
    dplyr::arrange(dplyr::desc(mean), std_err) |>
    dplyr::slice_head(n = 1L)

  if (nrow(best) != 1L || !".config" %in% names(best)) {
    stop("Could not identify one valid best configuration.")
  }
  best$.config[[1L]]
}

best_configs <- purrr::map_chr(results, get_best_config)

get_oof <- function(result, model_name) {
  pred <- tune::collect_predictions(result)
  if (!".pred_1" %in% names(pred)) {
    stop(model_name, ": .pred_1 is absent; check outcome factor levels.")
  }
  if (".config" %in% names(pred)) {
    pred <- dplyr::filter(pred, .config == best_configs[[model_name]])
  }

  out <- pred |>
    dplyr::group_by(.row) |>
    dplyr::summarise(
      success = dplyr::first(success),
      outcome_versions = dplyr::n_distinct(success),
      n_predictions = sum(is.finite(.pred_1)),
      .pred_1 = if (any(is.finite(.pred_1))) {
        mean(.pred_1[is.finite(.pred_1)])
      } else {
        NA_real_
      },
      .groups = "drop"
    ) |>
    dplyr::mutate(
      Case_ID = train_data$Case_ID[.row],
      Model = model_name,
      .before = 1L
    )

  if (nrow(out) != nrow(train_data) ||
      any(out$outcome_versions != 1L) ||
      any(out$n_predictions != CV_REPEATS) ||
      any(!is.finite(out$.pred_1)) ||
      any(out$.pred_1 < 0 | out$.pred_1 > 1)) {
    stop(model_name, ": incomplete or invalid OOF predictions.")
  }
  out
}

get_fold_auc <- function(result, model_name) {
  x <- tune::collect_metrics(result, summarize = FALSE) |>
    dplyr::filter(.metric == "roc_auc")

  if (".config" %in% names(x)) {
    x <- dplyr::filter(x, .config == best_configs[[model_name]])
  }

  if ("id2" %in% names(x)) {
    x <- dplyr::mutate(x, Repeat = as.character(id), Fold = as.character(id2))
  } else {
    x <- dplyr::mutate(x, Repeat = "Repeat1", Fold = as.character(id))
  }

  out <- x |>
    dplyr::transmute(
      Model = model_name,
      Repeat,
      Fold,
      Resample = paste(Repeat, Fold, sep = "_"),
      AUC = .estimate
    )

  if (nrow(out) != expected_resamples ||
      dplyr::n_distinct(out$Resample) != expected_resamples ||
      any(!is.finite(out$AUC))) {
    stop(model_name, ": incomplete or invalid fold-specific AUC values.")
  }
  out
}

oof_predictions <- purrr::imap_dfr(results, get_oof)
cv_fold_auc <- purrr::imap_dfr(results, get_fold_auc)

# Confirm that every algorithm used the same named resamples.
resample_sets <- split(cv_fold_auc$Resample, cv_fold_auc$Model)
if (!all(vapply(resample_sets[-1L], setequal, logical(1), resample_sets[[1L]]))) {
  stop("Algorithms do not have identical resample identifiers.")
}

collect_model_notes <- function(result, model_name) {
  x <- tune::collect_notes(result)
  if (nrow(x) == 0L) return(tibble::tibble())
  dplyr::mutate(x, Model = model_name, .before = 1L)
}

tuning_notes <- purrr::imap_dfr(results, collect_model_notes)
if (nrow(tuning_notes) > 0L) {
  warning(
    "Some candidate fits produced notes. Best configurations passed the strict ",
    "completeness checks, but review 07_tuning_notes.csv."
  )
}


# ==============================================================================
# 8. Summaries and model-selection review table
# ==============================================================================

message("6/8  Calculating summaries ...")

safe_calibration <- function(y, p) {
  lp <- stats::qlogis(pmin(pmax(p, 1e-6), 1 - 1e-6))

  intercept <- suppressWarnings(tryCatch(
    unname(stats::coef(stats::glm(
      y ~ 1 + offset(lp), family = stats::binomial()
    ))[1L]),
    error = function(e) NA_real_
  ))

  slope <- suppressWarnings(tryCatch(
    unname(stats::coef(stats::glm(
      y ~ lp, family = stats::binomial()
    ))[2L]),
    error = function(e) NA_real_
  ))

  c(intercept = intercept, slope = slope)
}

probability_summary <- function(data, model_name) {
  y <- as.integer(data$success == "1")
  p <- data$.pred_1
  cal <- safe_calibration(y, p)

  tibble::tibble(
    Model = model_name,
    OOF_AUC = yardstick::roc_auc_vec(
      data$success, p, event_level = "first"
    ),
    OOF_PR_AUC = yardstick::pr_auc_vec(
      data$success, p, event_level = "first"
    ),
    OOF_Brier = mean((p - y)^2),
    Calibration_intercept = unname(cal[["intercept"]]),
    Calibration_slope = unname(cal[["slope"]])
  )
}

get_youden <- function(data) {
  data |>
    yardstick::roc_curve(success, .pred_1, event_level = "first") |>
    dplyr::filter(is.finite(.threshold)) |>
    dplyr::mutate(Youden = sensitivity + specificity - 1) |>
    dplyr::arrange(dplyr::desc(Youden), dplyr::desc(sensitivity)) |>
    dplyr::slice_head(n = 1L) |>
    dplyr::pull(.threshold)
}

oof_performance <- oof_predictions |>
  dplyr::group_split(Model) |>
  purrr::map_dfr(~ probability_summary(.x, as.character(.x$Model[[1L]])))

thresholds <- oof_predictions |>
  dplyr::group_split(Model) |>
  purrr::map_dfr(~ tibble::tibble(
    Model = as.character(.x$Model[[1L]]),
    OOF_Youden_threshold = get_youden(.x)
  ))

note_counts <- if (nrow(tuning_notes) == 0L) {
  tibble::tibble(Model = character(), Tuning_notes_n = integer())
} else {
  tuning_notes |>
    dplyr::count(Model, name = "Tuning_notes_n")
}

cv_summary <- cv_fold_auc |>
  dplyr::group_by(Model) |>
  dplyr::summarise(
    n_resamples = dplyr::n(),
    CV_AUC_mean = mean(AUC),
    CV_AUC_SD = stats::sd(AUC),
    CV_AUC_median = stats::median(AUC),
    CV_AUC_min = min(AUC),
    CV_AUC_max = max(AUC),
    .groups = "drop"
  ) |>
  dplyr::left_join(oof_performance, by = "Model") |>
  dplyr::left_join(thresholds, by = "Model") |>
  dplyr::left_join(note_counts, by = "Model") |>
  dplyr::mutate(
    Tuning_notes_n = tidyr::replace_na(Tuning_notes_n, 0L),
    Complete = n_resamples == expected_resamples & is.finite(CV_AUC_mean)
  )

best_mean_auc <- max(cv_summary$CV_AUC_mean[cv_summary$Complete])

model_selection_review <- cv_summary |>
  dplyr::mutate(
    AUC_delta_from_best = best_mean_auc - CV_AUC_mean,
    Within_0.01_of_best = Complete &
      AUC_delta_from_best <= AUC_NEAR_BEST_MARGIN + sqrt(.Machine$double.eps)
  ) |>
  dplyr::arrange(
    dplyr::desc(Complete),
    dplyr::desc(CV_AUC_mean),
    CV_AUC_SD,
    OOF_Brier,
    Model
  ) |>
  dplyr::mutate(
    CV_rank = dplyr::row_number(),
    Model_selected = FALSE,
    Selection_status = dplyr::case_when(
      !Complete ~ "Ineligible: incomplete best-configuration resamples",
      AUC_delta_from_best < sqrt(.Machine$double.eps) ~
        "Largest mean CV AUC; manual confirmation required",
      Within_0.01_of_best ~
        "Near-best AUC; compare stability, Brier, calibration and deployability",
      TRUE ~ "Below the prespecified near-best AUC range"
    )
  )

best_parameters <- purrr::imap_dfr(results, function(result, model_name) {
  if (is.null(grids[[model_name]])) {
    return(tibble::tibble(
      Model = model_name, Parameter = "No tuning", Value = NA_character_
    ))
  }

  tune::select_best(result, metric = "roc_auc") |>
    dplyr::select(-dplyr::any_of(".config")) |>
    dplyr::mutate(dplyr::across(dplyr::everything(), as.character)) |>
    tidyr::pivot_longer(
      dplyr::everything(), names_to = "Parameter", values_to = "Value"
    ) |>
    dplyr::mutate(Model = model_name, .before = 1L)
})


# ==============================================================================
# 9. Save training-CV outputs only (never use save.image())
# ==============================================================================

message("7/8  Saving development-CV outputs ...")

readr::write_csv(cv_fold_auc,
                 file.path(output_dir, "01_development_CV_AUC_values.csv"))
readr::write_csv(cv_summary,
                 file.path(output_dir, "02_development_CV_model_summary.csv"))
readr::write_csv(best_parameters,
                 file.path(output_dir, "03_best_hyperparameters.csv"))
readr::write_csv(thresholds,
                 file.path(output_dir, "04_development_OOF_Youden_thresholds.csv"))
readr::write_csv(model_selection_review,
                 file.path(output_dir, "05_model_selection_FOR_MANUAL_REVIEW.csv"))

run_configuration <- tibble::tribble(
  ~Item, ~Value,
  "Run_mode", ifelse(QUICK_MODE, "QUICK_CODE_CHECK", "FORMAL_CV"),
  "Development_source", basename(training_file),
  "Development_source_MD5", training_file_md5,
  "Development_n", as.character(EXPECTED_DEVELOPMENT_N),
  "Development_success_n", as.character(EXPECTED_SUCCESS_N),
  "Development_failure_n", as.character(EXPECTED_FAILURE_N),
  "Sealed_internal_test_n", as.character(EXPECTED_TEST_N),
  "CV_folds", as.character(CV_FOLDS),
  "CV_repeats", as.character(CV_REPEATS),
  "CV_seed", as.character(SEED_CV),
  "Feature_selection", "Conventional Boruta in the 621-case development cohort",
  "CV_interpretation", "Post-Boruta algorithm comparison",
  "Positive_event", "Success = 1 = first factor level",
  "Probability_column", ".pred_1",
  "Threshold_source", "Mean repeated-CV OOF predictions in development cohort",
  "Heldout_cohort_extracted_or_evaluated", "FALSE",
  "Automatic_model_selection", "FALSE",
  "Near_best_AUC_margin", as.character(AUC_NEAR_BEST_MARGIN)
)

readr::write_csv(run_configuration,
                 file.path(output_dir, "06_run_configuration.csv"))
readr::write_csv(tuning_notes,
                 file.path(output_dir, "07_tuning_notes.csv"))
readr::write_csv(tibble::tibble(Case_ID = development_ids),
                 file.path(output_dir, "08_development_row_ids.csv"))
readr::write_csv(oof_predictions,
                 file.path(output_dir, "09_development_OOF_predictions.csv"))
readr::write_csv(missingness,
                 file.path(output_dir, "10_development_missingness.csv"))

# This object contains development-CV data/results only. It does not contain
# train_raw or any internal-test data.
analysis_objects <- list(
  run_configuration = run_configuration,
  development_row_ids = development_ids,
  cv_folds = cv_folds,
  workflows = workflows,
  grids = grids,
  results = results,
  best_configs = best_configs,
  best_parameters = best_parameters,
  cv_fold_auc = cv_fold_auc,
  cv_summary = cv_summary,
  model_selection_review = model_selection_review,
  oof_predictions = oof_predictions,
  thresholds = thresholds
)

saveRDS(
  analysis_objects,
  file.path(output_dir, "11_ESWL_9_models_development_CV_objects.rds")
)

writeLines(
  capture.output(utils::sessionInfo()),
  file.path(output_dir, "12_sessionInfo.txt")
)


# ==============================================================================
# 10. Console summary
# ==============================================================================

message("8/8  Completed. No final model has been selected automatically.")
print(model_selection_review)

cat(
  "\nManual decision rule:\n",
  "1) require complete results for all ", expected_resamples, " resamples;\n",
  "2) use mean CV ROC AUC as the primary criterion;\n",
  "3) when AUC differs by <= ", AUC_NEAR_BEST_MARGIN,
  ", compare CV SD, OOF Brier/calibration and deployability;\n",
  "4) lock one algorithm before opening the 208-case internal test set.\n",
  "\nResults saved to: ", output_dir, "\n",
  sep = ""
)
