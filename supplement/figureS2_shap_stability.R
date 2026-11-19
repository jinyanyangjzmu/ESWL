# =============================================================================
# Publication-grade SHAP consistency analysis for the locked 6-variable
# XGBoost model predicting composite first-session ESWL success
#
# Outputs:
#   1) Five-fold CV SHAP stability curves (recommended Supplementary Figure S2)
#   2) CV importance-rank and curve-concordance summary
#   3) Locked-model SHAP consistency: training vs internal test cohort
#
# Important design rules:
#   - Cross-validation is performed only in the 621-patient training cohort.
#   - The locked recipe and locked hyperparameters are reused in every fold.
#   - No hyperparameter tuning or feature selection is performed here.
#   - The 208-patient internal test cohort is used only after the CV analysis.
#   - All SHAP values explain P(composite ESWL success), not failure.
# =============================================================================

# Install once if required:
# install.packages(c(
#   "tidymodels", "fastshap", "ggplot2", "patchwork", "dplyr",
#   "tidyr", "readr", "scales"
# ))

suppressPackageStartupMessages({
  library(tidymodels)
  library(fastshap)
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(scales)
})

# =============================================================================
# 1. Configuration: edit only this section
# =============================================================================

model_file <- 'output/model_benchmark_9_CV_FORMAL/13_final_models/04_XGBoost_best_CV_refit_621.rds'
training_file <- 'output/feature_selection/01_traindata.rds'
internal_test_file <- "output/feature_selection/01_testdata_SEALED.rds"

output_dir <- "output/SHAP_consistency_6variable_XGBoost"

cv_seed <- 20260904L
cv_folds <- 5L

# Final publication run: 200 or more Monte Carlo repetitions.
# For a quick code check only, temporarily use 20.
cv_shap_nsim <- 200L
final_model_shap_nsim <- 300L

# Set FALSE only if the training-vs-internal-test figure is not required.
run_internal_test_consistency <- TRUE

for (required_file in c(model_file, training_file, internal_test_file)) {
  if (!file.exists(required_file)) {
    stop("Required file not found: ", normalizePath(required_file, mustWork = FALSE))
  }
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
base_family <- if (.Platform$OS.type == "windows") "Arial" else "sans"

# =============================================================================
# 2. Reconstruct the exact six-variable raw-data interface
# =============================================================================

locked_xgboost <- readRDS(model_file)
training_raw <- readRDS(training_file)
internal_test_raw <- readRDS(internal_test_file)

required_source_columns <- c(
  "Stone volume",
  "Mean CT attenuation",
  "Ureteral wall thickness",
  "Degree of hydronephrosis",
  "Subcutaneous fat thickness",
  "Stone location",
  "success"
)

check_source_columns <- function(data, dataset_name) {
  missing_columns <- setdiff(required_source_columns, names(data))
  if (length(missing_columns) > 0L) {
    stop(
      dataset_name, " is missing required columns: ",
      paste(missing_columns, collapse = ", ")
    )
  }
}

check_source_columns(training_raw, "Training data")
check_source_columns(internal_test_raw, "Internal test data")

stone_location_levels <- c(
  "Distal ureter",
  "Mid ureter",
  "Proximal ureter"
)

prepare_model_data <- function(data, id_offset = 0L) {
  outcome_character <- as.character(data[["success"]])

  # Accept either No/Yes or 0/1 source coding and initially store a neutral
  # clinical 0/1 representation. Before CV refitting, this factor is recast to
  # the exact labels, order, and ordered attribute required by the locked recipe.
  outcome_01 <- dplyr::case_when(
    outcome_character %in% c("Yes", "1") ~ "1",
    outcome_character %in% c("No", "0") ~ "0",
    TRUE ~ NA_character_
  )

  prepared <- tibble(
    Case_ID = id_offset + seq_len(nrow(data)),
    Stone_volume = as.numeric(data[["Stone volume"]]),
    Mean_CT_attenuation = as.numeric(data[["Mean CT attenuation"]]),
    Ureteral_wall_thickness = as.numeric(data[["Ureteral wall thickness"]]),
    Hydronephrosis_measure = as.numeric(data[["Degree of hydronephrosis"]]),
    Subcutaneous_fat_thickness = as.numeric(
      data[["Subcutaneous fat thickness"]]
    ),
    Stone_location = factor(
      as.character(data[["Stone location"]]),
      levels = stone_location_levels
    ),
    success = factor(outcome_01, levels = c("0", "1"))
  )

  if (anyNA(prepared)) {
    na_counts <- vapply(prepared, function(z) sum(is.na(z)), integer(1))
    stop(
      "Missing or unrecognized values after data reconstruction: ",
      paste(
        names(na_counts)[na_counts > 0L],
        na_counts[na_counts > 0L],
        sep = "=",
        collapse = ", "
      )
    )
  }

  prepared
}

training_data <- prepare_model_data(training_raw, id_offset = 0L)
internal_test_data <- prepare_model_data(
  internal_test_raw,
  id_offset = 100000L
)

if (nrow(training_data) != 621L) {
  warning("Expected 621 training patients but found ", nrow(training_data), ".")
}
if (nrow(internal_test_data) != 208L) {
  warning(
    "Expected 208 internal-test patients but found ",
    nrow(internal_test_data), "."
  )
}

predictor_names <- c(
  "Stone_volume",
  "Mean_CT_attenuation",
  "Ureteral_wall_thickness",
  "Hydronephrosis_measure",
  "Subcutaneous_fat_thickness",
  "Stone_location"
)

continuous_predictors <- predictor_names[
  predictor_names != "Stone_location"
]

feature_labels <- c(
  Stone_volume = "Stone volume (mm\u00b3)",
  Mean_CT_attenuation = "Mean CT attenuation (HU)",
  Ureteral_wall_thickness = "Ureteral wall thickness (mm)",
  Hydronephrosis_measure = "Hydronephrosis measurement (mm)",
  Subcutaneous_fat_thickness = "Subcutaneous fat thickness (mm)",
  Stone_location = "Stone location"
)

feature_labels_short <- c(
  Stone_volume = "SV (mm\u00b3)",
  Mean_CT_attenuation = "MeCT (HU)",
  Ureteral_wall_thickness = "UWT (mm)",
  Hydronephrosis_measure = "Hydronephrosis (mm)",
  Subcutaneous_fat_thickness = "SFT (mm)",
  Stone_location = "Stone location"
)

# =============================================================================
# 3. Locked workflow checks and extraction
# =============================================================================

probe_prediction <- predict(
  locked_xgboost,
  new_data = training_data,
  type = "prob"
)

success_probability_candidates <- c(".pred_1", ".pred_Yes")
available_success_columns <- intersect(
  success_probability_candidates,
  names(probe_prediction)
)

if (length(available_success_columns) != 1L) {
  stop(
    "Could not identify the success-probability column. Available columns: ",
    paste(names(probe_prediction), collapse = ", ")
  )
}

locked_target_col <- available_success_columns[[1L]]
message("Locked-model success probability: ", locked_target_col)

locked_training_probability <- as.numeric(
  probe_prediction[[locked_target_col]]
)

mean_probability_by_outcome <- tapply(
  locked_training_probability,
  training_data$success,
  mean
)

if (unname(mean_probability_by_outcome["1"]) <=
    unname(mean_probability_by_outcome["0"])) {
  stop(
    "Probability-direction audit failed: the selected probability is not ",
    "higher among observed successes."
  )
}

message(sprintf(
  "Direction audit passed: mean P(success) = %.3f for success and %.3f for failure.",
  unname(mean_probability_by_outcome["1"]),
  unname(mean_probability_by_outcome["0"])
))

# Extract the finalized XGBoost specification. This reuses exactly the locked
# hyperparameters but does not reuse the full-data fitted booster in CV folds.
locked_parsnip_fit <- workflows::extract_fit_parsnip(locked_xgboost)
locked_model_spec <- locked_parsnip_fit$spec

if (is.null(locked_model_spec)) {
  stop("Could not extract the finalized parsnip model specification.")
}

# Prefer the exact unestimated recipe stored in the locked workflow. The
# fallback reproduces the confirmed recipe structure from the fitted object.
locked_recipe <- tryCatch(
  workflows::extract_recipe(locked_xgboost, estimated = FALSE),
  error = function(e) NULL
)

if (is.null(locked_recipe)) {
  warning(
    "The unestimated recipe could not be extracted; using the confirmed ",
    "six-variable recipe structure."
  )

  locked_recipe <- recipe(success ~ ., data = training_data) %>%
    update_role(Case_ID, new_role = "case_id") %>%
    step_novel(all_nominal_predictors(), new_level = "__NEW__") %>%
    step_dummy(all_nominal_predictors()) %>%
    step_zv(all_predictors())
}

# -----------------------------------------------------------------------------
# Harmonize the outcome with the exact recipe prototype
# -----------------------------------------------------------------------------
# Predicting from a fitted workflow does not require the outcome column, so an
# outcome-label mismatch can remain invisible during ordinary prediction.
# Refitting a recipe inside CV is stricter: factor labels, level order, and the
# ordered attribute must match the outcome prototype stored in the recipe.
recipe_prototype <- recipes::recipes_ptype(locked_recipe)

if (!"success" %in% names(recipe_prototype)) {
  stop("The locked recipe prototype does not contain the outcome 'success'.")
}

expected_success_prototype <- recipe_prototype$success
expected_success_levels <- levels(expected_success_prototype)
expected_success_ordered <- is.ordered(expected_success_prototype)

if (is.null(expected_success_levels) || length(expected_success_levels) != 2L) {
  stop(
    "The locked recipe outcome must be a two-level factor. Detected levels: ",
    paste(expected_success_levels, collapse = ", ")
  )
}

message(
  "Locked recipe outcome levels: ",
  paste(expected_success_levels, collapse = " -> "),
  if (expected_success_ordered) " (ordered factor)" else " (unordered factor)"
)

recast_success_to_recipe <- function(success_01) {
  clinical_01 <- as.character(success_01)

  if (setequal(expected_success_levels, c("0", "1"))) {
    mapped_values <- clinical_01
    success_event_level <- "1"
  } else if (setequal(expected_success_levels, c("No", "Yes"))) {
    mapped_values <- ifelse(clinical_01 == "1", "Yes", "No")
    success_event_level <- "Yes"
  } else if (setequal(tolower(expected_success_levels), c("no", "yes"))) {
    no_level <- expected_success_levels[tolower(expected_success_levels) == "no"]
    yes_level <- expected_success_levels[tolower(expected_success_levels) == "yes"]
    mapped_values <- ifelse(clinical_01 == "1", yes_level, no_level)
    success_event_level <- yes_level
  } else {
    stop(
      "Unsupported outcome levels in the locked recipe: ",
      paste(expected_success_levels, collapse = ", "),
      ". Expected 0/1 or No/Yes coding."
    )
  }

  recast_factor <- factor(
    mapped_values,
    levels = expected_success_levels,
    ordered = expected_success_ordered
  )

  list(
    values = recast_factor,
    success_event_level = success_event_level
  )
}

training_success_recast <- recast_success_to_recipe(training_data$success)
internal_test_success_recast <- recast_success_to_recipe(
  internal_test_data$success
)

training_data$success <- training_success_recast$values
internal_test_data$success <- internal_test_success_recast$values
success_event_level <- training_success_recast$success_event_level

if (!identical(attributes(training_data$success),
               attributes(expected_success_prototype))) {
  # Zero-length recipe prototypes can carry names that patient-level vectors
  # do not. Compare only the factor-defining attributes relevant to prep().
  same_levels <- identical(
    levels(training_data$success),
    levels(expected_success_prototype)
  )
  same_ordered <- identical(
    is.ordered(training_data$success),
    is.ordered(expected_success_prototype)
  )
  same_class <- identical(
    class(training_data$success),
    class(expected_success_prototype)
  )

  if (!all(same_levels, same_ordered, same_class)) {
    stop(
      "Outcome harmonization failed: the reconstructed factor still does not ",
      "match the locked recipe prototype."
    )
  }
}

message(
  "CV outcome harmonized successfully; clinical success level = '",
  success_event_level,
  "'."
)

cv_workflow_spec <- workflow() %>%
  add_recipe(locked_recipe) %>%
  add_model(locked_model_spec)

# =============================================================================
# 4. Prediction and FastSHAP helper functions
# =============================================================================

prepare_predictor_frame <- function(data) {
  data %>%
    select(all_of(predictor_names)) %>%
    as.data.frame()
}

prepare_newdata_for_workflow <- function(newdata) {
  newdata <- as.data.frame(newdata)

  missing_predictors <- setdiff(predictor_names, names(newdata))
  if (length(missing_predictors) > 0L) {
    stop(
      "Prediction data are missing: ",
      paste(missing_predictors, collapse = ", ")
    )
  }

  newdata$Stone_location <- factor(
    as.character(newdata$Stone_location),
    levels = stone_location_levels
  )

  if (anyNA(newdata$Stone_location)) {
    stop("Unrecognized Stone_location level encountered during prediction.")
  }

  newdata$Case_ID <- seq_len(nrow(newdata))
  newdata %>% select(Case_ID, all_of(predictor_names))
}

predict_success_probability <- function(object, newdata) {
  prediction_data <- prepare_newdata_for_workflow(newdata)
  prediction <- predict(object, new_data = prediction_data, type = "prob")

  candidate <- intersect(c(".pred_1", ".pred_Yes"), names(prediction))
  if (length(candidate) != 1L) {
    stop(
      "Exactly one success-probability column was expected; returned: ",
      paste(names(prediction), collapse = ", ")
    )
  }

  as.numeric(prediction[[candidate[[1L]]]])
}

calculate_probability_shap <- function(
    fitted_workflow,
    background_X,
    explained_X,
    nsim,
    seed
) {
  set.seed(seed)

  shap_matrix <- fastshap::explain(
    object = fitted_workflow,
    X = background_X,
    newdata = explained_X,
    pred_wrapper = predict_success_probability,
    nsim = nsim,
    adjust = TRUE,
    shap_only = TRUE
  )

  stopifnot(
    nrow(shap_matrix) == nrow(explained_X),
    identical(colnames(shap_matrix), colnames(explained_X))
  )

  baseline <- attr(shap_matrix, "baseline")
  probability <- predict_success_probability(fitted_workflow, explained_X)
  reconstructed <- baseline + rowSums(shap_matrix)
  max_error <- max(abs(probability - reconstructed))

  list(
    S = shap_matrix,
    baseline = baseline,
    probability = probability,
    max_local_accuracy_error = max_error
  )
}

shap_to_long <- function(shap_result, X, fold, subset_name) {
  bind_rows(lapply(predictor_names, function(variable) {
    is_continuous <- variable %in% continuous_predictors

    tibble(
      CV_fold = factor(fold, levels = seq_len(cv_folds)),
      subset = factor(
        subset_name,
        levels = c("Analysis", "Assessment")
      ),
      row_in_subset = seq_len(nrow(X)),
      feature_code = variable,
      feature = unname(feature_labels[variable]),
      feature_value_numeric = if (is_continuous) {
        as.numeric(X[[variable]])
      } else {
        NA_real_
      },
      feature_value_category = if (!is_continuous) {
        as.character(X[[variable]])
      } else {
        NA_character_
      },
      shap_value = as.numeric(shap_result$S[, variable]),
      predicted_success_probability = shap_result$probability,
      SHAP_baseline = shap_result$baseline
    )
  }))
}

# =============================================================================
# 5. Five-fold CV SHAP consistency analysis in the training cohort only
# =============================================================================

set.seed(cv_seed)
cv_resamples <- rsample::vfold_cv(
  training_data,
  v = cv_folds,
  strata = success
)

cv_shap_long_list <- vector("list", cv_folds * 2L)
cv_accuracy_audit_list <- vector("list", cv_folds * 2L)
cv_fold_sizes_list <- vector("list", cv_folds)

list_position <- 1L

for (fold_index in seq_len(cv_folds)) {
  message("Fitting locked XGBoost specification in CV fold ", fold_index, "...")

  analysis_data <- rsample::analysis(cv_resamples$splits[[fold_index]])
  assessment_data <- rsample::assessment(cv_resamples$splits[[fold_index]])

  set.seed(cv_seed + fold_index)
  fold_workflow <- fit(cv_workflow_spec, data = analysis_data)

  X_analysis <- prepare_predictor_frame(analysis_data)
  X_assessment <- prepare_predictor_frame(assessment_data)

  # Each fold's analysis subset is the background distribution for both the
  # analysis and assessment SHAP calculations in that fold.
  shap_analysis <- calculate_probability_shap(
    fitted_workflow = fold_workflow,
    background_X = X_analysis,
    explained_X = X_analysis,
    nsim = cv_shap_nsim,
    seed = cv_seed + fold_index * 100L + 1L
  )

  shap_assessment <- calculate_probability_shap(
    fitted_workflow = fold_workflow,
    background_X = X_analysis,
    explained_X = X_assessment,
    nsim = cv_shap_nsim,
    seed = cv_seed + fold_index * 100L + 2L
  )

  cv_shap_long_list[[list_position]] <- shap_to_long(
    shap_analysis,
    X_analysis,
    fold_index,
    "Analysis"
  )
  cv_accuracy_audit_list[[list_position]] <- tibble(
    CV_fold = fold_index,
    subset = "Analysis",
    n = nrow(X_analysis),
    SHAP_baseline = shap_analysis$baseline,
    max_local_accuracy_error = shap_analysis$max_local_accuracy_error
  )
  list_position <- list_position + 1L

  cv_shap_long_list[[list_position]] <- shap_to_long(
    shap_assessment,
    X_assessment,
    fold_index,
    "Assessment"
  )
  cv_accuracy_audit_list[[list_position]] <- tibble(
    CV_fold = fold_index,
    subset = "Assessment",
    n = nrow(X_assessment),
    SHAP_baseline = shap_assessment$baseline,
    max_local_accuracy_error = shap_assessment$max_local_accuracy_error
  )
  list_position <- list_position + 1L

  cv_fold_sizes_list[[fold_index]] <- tibble(
    CV_fold = fold_index,
    analysis_n = nrow(analysis_data),
    assessment_n = nrow(assessment_data),
    analysis_success_n = sum(analysis_data$success == success_event_level),
    assessment_success_n = sum(assessment_data$success == success_event_level)
  )

  rm(fold_workflow, shap_analysis, shap_assessment)
  invisible(gc())
}

cv_shap_long <- bind_rows(cv_shap_long_list)
cv_accuracy_audit <- bind_rows(cv_accuracy_audit_list)
cv_fold_sizes <- bind_rows(cv_fold_sizes_list)

if (max(cv_accuracy_audit$max_local_accuracy_error) > 1e-6) {
  warning(
    "At least one adjusted FastSHAP run has local-accuracy error > 1e-6. ",
    "Inspect CV_SHAP_local_accuracy_audit.csv."
  )
}

# =============================================================================
# 6. Quantitative CV consistency summaries
# =============================================================================

cv_importance <- cv_shap_long %>%
  group_by(CV_fold, subset, feature_code, feature) %>%
  summarise(
    mean_absolute_SHAP = mean(abs(shap_value)),
    .groups = "drop"
  ) %>%
  group_by(CV_fold, subset) %>%
  mutate(
    importance_rank = min_rank(desc(mean_absolute_SHAP))
  ) %>%
  ungroup()

importance_rank_concordance <- cv_importance %>%
  select(CV_fold, subset, feature_code, importance_rank) %>%
  pivot_wider(
    names_from = subset,
    values_from = importance_rank
  ) %>%
  group_by(CV_fold) %>%
  summarise(
    Spearman_rank_rho = cor(
      Analysis,
      Assessment,
      method = "spearman",
      use = "complete.obs"
    ),
    .groups = "drop"
  )

calculate_curve_concordance <- function(data_long, feature_code) {
  feature_data <- data_long %>% filter(.data$feature_code == feature_code)

  if (feature_code %in% continuous_predictors) {
    # Common training-cohort quantile bins permit like-for-like comparison of
    # each fold's analysis and assessment effect pattern.
    source_values <- training_data[[feature_code]]
    break_points <- unique(as.numeric(quantile(
      source_values,
      probs = seq(0, 1, length.out = 11L),
      na.rm = TRUE,
      names = FALSE
    )))

    if (length(break_points) < 4L) {
      return(tibble())
    }

    break_points[1] <- -Inf
    break_points[length(break_points)] <- Inf

    summarized <- feature_data %>%
      mutate(
        comparison_bin = cut(
          feature_value_numeric,
          breaks = break_points,
          include.lowest = TRUE,
          labels = FALSE
        )
      ) %>%
      group_by(CV_fold, subset, comparison_bin) %>%
      summarise(
        mean_SHAP = mean(shap_value),
        n_in_bin = n(),
        .groups = "drop"
      )
  } else {
    summarized <- feature_data %>%
      mutate(comparison_bin = feature_value_category) %>%
      group_by(CV_fold, subset, comparison_bin) %>%
      summarise(
        mean_SHAP = mean(shap_value),
        n_in_bin = n(),
        .groups = "drop"
      )
  }

  summarized %>%
    select(CV_fold, subset, comparison_bin, mean_SHAP) %>%
    pivot_wider(names_from = subset, values_from = mean_SHAP) %>%
    group_by(CV_fold) %>%
    summarise(
      feature_code = feature_code,
      n_common_bins = sum(complete.cases(Analysis, Assessment)),
      curve_Spearman_rho = if (
        sum(complete.cases(Analysis, Assessment)) >= 3L
      ) {
        cor(
          Analysis,
          Assessment,
          method = "spearman",
          use = "complete.obs"
        )
      } else {
        NA_real_
      },
      curve_RMSE = sqrt(mean(
        (Analysis - Assessment)^2,
        na.rm = TRUE
      )),
      .groups = "drop"
    )
}

cv_curve_concordance <- bind_rows(lapply(
  predictor_names,
  function(variable) {
    calculate_curve_concordance(cv_shap_long, variable)
  }
)) %>%
  mutate(feature = unname(feature_labels_short[feature_code]))

write_csv(
  cv_fold_sizes,
  file.path(output_dir, "CV_fold_sizes_and_outcomes.csv")
)
write_csv(
  cv_accuracy_audit,
  file.path(output_dir, "CV_SHAP_local_accuracy_audit.csv")
)
write_csv(
  cv_importance,
  file.path(output_dir, "CV_SHAP_importance_and_ranks.csv")
)
write_csv(
  importance_rank_concordance,
  file.path(output_dir, "CV_SHAP_importance_rank_concordance.csv")
)
write_csv(
  cv_curve_concordance,
  file.path(output_dir, "CV_SHAP_curve_concordance.csv")
)
write_csv(
  cv_shap_long,
  file.path(output_dir, "CV_SHAP_probability_scale_long.csv")
)

# =============================================================================
# 7. Publication theme and CV stability figure
# =============================================================================

fold_colors <- c(
  "1" = "#0072B2",
  "2" = "#D55E00",
  "3" = "#009E73",
  "4" = "#CC79A7",
  "5" = "#E69F00"
)

theme_publication <- function(base_size = 8.2) {
  theme_bw(base_size = base_size, base_family = base_family) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.22, colour = "#EBEBEB"),
      panel.border = element_rect(linewidth = 0.5, colour = "#333333"),
      axis.title = element_text(colour = "#111111"),
      axis.text = element_text(colour = "#333333"),
      plot.title = element_text(face = "bold", size = rel(1.05), hjust = 0),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = rel(0.95)),
      strip.placement = "outside",
      legend.title = element_text(face = "bold"),
      plot.margin = margin(4, 5, 4, 4)
    )
}

make_cv_continuous_plot <- function(variable) {
  plot_data <- cv_shap_long %>%
    filter(feature_code == variable)

  display_limits <- as.numeric(quantile(
    training_data[[variable]],
    probs = c(0.01, 0.99),
    na.rm = TRUE,
    names = FALSE
  ))

  ggplot(
    plot_data,
    aes(
      x = feature_value_numeric,
      y = shap_value,
      colour = CV_fold,
      group = CV_fold
    )
  ) +
    geom_hline(
      yintercept = 0,
      linewidth = 0.4,
      linetype = "dashed",
      colour = "#686868"
    ) +
    geom_point(
      colour = "#AFAFAF",
      size = 0.28,
      alpha = 0.10,
      show.legend = FALSE
    ) +
    geom_smooth(
      method = "loess",
      formula = y ~ x,
      span = 0.80,
      se = FALSE,
      linewidth = 0.62,
      na.rm = TRUE
    ) +
    facet_grid(. ~ subset) +
    coord_cartesian(xlim = display_limits) +
    scale_colour_manual(values = fold_colors, drop = FALSE) +
    labs(
      title = unname(feature_labels[variable]),
      x = unname(feature_labels_short[variable]),
      y = "SHAP value for P(success)",
      colour = "CV fold"
    ) +
    theme_publication(8.2) +
    theme(legend.position = "bottom")
}

make_cv_categorical_plot <- function(variable) {
  plot_data <- cv_shap_long %>%
    filter(feature_code == variable)

  summary_data <- plot_data %>%
    group_by(CV_fold, subset, feature_value_category) %>%
    summarise(
      mean_SHAP = mean(shap_value),
      SE = sd(shap_value) / sqrt(n()),
      lower = mean_SHAP - 1.96 * SE,
      upper = mean_SHAP + 1.96 * SE,
      .groups = "drop"
    ) %>%
    mutate(
      feature_value_category = factor(
        feature_value_category,
        levels = stone_location_levels
      )
    )

  dodge <- position_dodge(width = 0.58)

  ggplot(
    summary_data,
    aes(
      x = feature_value_category,
      y = mean_SHAP,
      colour = CV_fold,
      group = CV_fold
    )
  ) +
    geom_hline(
      yintercept = 0,
      linewidth = 0.4,
      linetype = "dashed",
      colour = "#686868"
    ) +
    geom_errorbar(
      aes(ymin = lower, ymax = upper),
      position = dodge,
      width = 0.14,
      linewidth = 0.45
    ) +
    geom_point(
      position = dodge,
      size = 1.35
    ) +
    facet_grid(. ~ subset) +
    scale_colour_manual(values = fold_colors, drop = FALSE) +
    scale_x_discrete(labels = c(
      "Distal ureter" = "Distal",
      "Mid ureter" = "Mid",
      "Proximal ureter" = "Proximal"
    )) +
    labs(
      title = unname(feature_labels[variable]),
      x = "Ureteral stone location",
      y = "SHAP value for P(success)",
      colour = "CV fold"
    ) +
    theme_publication(8.2) +
    theme(legend.position = "bottom")
}

cv_feature_plots <- lapply(predictor_names, function(variable) {
  if (variable %in% continuous_predictors) {
    make_cv_continuous_plot(variable)
  } else {
    make_cv_categorical_plot(variable)
  }
})

figure_cv_stability <- patchwork::wrap_plots(
  cv_feature_plots,
  ncol = 2,
  guides = "collect"
) +
  patchwork::plot_annotation(
    tag_levels = "A",
    theme = theme(
      plot.tag = element_text(
        family = base_family,
        face = "bold",
        size = 11
      ),
      plot.caption = element_text(
        family = base_family,
        size = 6.5,
        colour = "#555555",
        hjust = 0,
        margin = margin(t = 5)
      ),
      legend.position = "bottom"
    )
  )

# =============================================================================
# 8. CV consistency summary figure
# =============================================================================

median_rank_order <- cv_importance %>%
  group_by(feature_code) %>%
  summarise(median_rank = median(importance_rank), .groups = "drop") %>%
  arrange(desc(median_rank)) %>%
  pull(feature_code)

rank_plot_data <- cv_importance %>%
  mutate(
    feature_short = factor(
      unname(feature_labels_short[feature_code]),
      levels = unname(feature_labels_short[median_rank_order])
    ),
    fold_subset = factor(
      paste0(
        "F", CV_fold, "\n",
        ifelse(subset == "Analysis", "A", "H")
      ),
      levels = as.vector(rbind(
        paste0("F", seq_len(cv_folds), "\nA"),
        paste0("F", seq_len(cv_folds), "\nH")
      ))
    )
  )

p_rank <- ggplot(
  rank_plot_data,
  aes(x = fold_subset, y = feature_short, fill = importance_rank)
) +
  geom_tile(colour = "white", linewidth = 0.45) +
  geom_text(
    aes(label = importance_rank),
    family = base_family,
    size = 2.45,
    colour = "#111111"
  ) +
  scale_fill_gradient(
    low = "#D73027",
    high = "#4575B4",
    limits = c(1, length(predictor_names)),
    breaks = seq_len(length(predictor_names)),
    name = "Importance\nrank"
  ) +
  labs(
    title = "Feature-importance rank stability",
    subtitle = "A = analysis subset; H = held-out assessment subset",
    x = "CV fold and subset",
    y = NULL
  ) +
  theme_publication(8.2) +
  theme(panel.grid = element_blank())

curve_plot_data <- cv_curve_concordance %>%
  mutate(
    feature_short = factor(
      feature,
      levels = rev(unname(feature_labels_short[predictor_names]))
    )
  )

p_curve_rho <- ggplot(
  curve_plot_data,
  aes(
    x = curve_Spearman_rho,
    y = feature_short,
    colour = CV_fold
  )
) +
  geom_vline(
    xintercept = 0,
    linewidth = 0.4,
    linetype = "dashed",
    colour = "#777777"
  ) +
  geom_point(size = 1.7, alpha = 0.90) +
  scale_colour_manual(values = fold_colors, drop = FALSE) +
  scale_x_continuous(
    limits = c(-1, 1),
    breaks = seq(-1, 1, by = 0.5)
  ) +
  labs(
    title = "Analysis-assessment SHAP curve concordance",
    subtitle = "Spearman correlation of binned mean SHAP patterns",
    x = "Spearman correlation",
    y = NULL,
    colour = "CV fold"
  ) +
  theme_publication(8.2) +
  theme(legend.position = "bottom")

figure_cv_summary <- p_rank / p_curve_rho +
  patchwork::plot_annotation(
    tag_levels = "A",
    caption = paste0(
      "For continuous predictors, concordance was calculated across ten common ",
      "training-cohort quantile bins. Stone-location correlations are based on ",
      "three category-specific mean SHAP values and should be interpreted cautiously."
    ),
    theme = theme(
      plot.tag = element_text(
        family = base_family,
        face = "bold",
        size = 11
      ),
      plot.caption = element_text(
        family = base_family,
        size = 6.5,
        colour = "#555555",
        hjust = 0
      )
    )
  )

# =============================================================================
# 9. Locked final model: training vs internal-test SHAP consistency
# =============================================================================

final_consistency_long <- NULL
final_curve_concordance <- NULL
figure_train_test <- NULL

if (isTRUE(run_internal_test_consistency)) {
  message("Calculating locked-model SHAP values in training and internal test cohorts...")

  X_training_final <- prepare_predictor_frame(training_data)
  X_internal_test_final <- prepare_predictor_frame(internal_test_data)

  # The full training cohort is the common reference distribution for both.
  final_shap_training <- calculate_probability_shap(
    fitted_workflow = locked_xgboost,
    background_X = X_training_final,
    explained_X = X_training_final,
    nsim = final_model_shap_nsim,
    seed = cv_seed + 9001L
  )

  final_shap_internal_test <- calculate_probability_shap(
    fitted_workflow = locked_xgboost,
    background_X = X_training_final,
    explained_X = X_internal_test_final,
    nsim = final_model_shap_nsim,
    seed = cv_seed + 9002L
  )

  final_shap_to_long <- function(shap_result, X, cohort_name) {
    bind_rows(lapply(predictor_names, function(variable) {
      is_continuous <- variable %in% continuous_predictors

      tibble(
        cohort = factor(
          cohort_name,
          levels = c("Training", "Internal test")
        ),
        row_in_cohort = seq_len(nrow(X)),
        feature_code = variable,
        feature_value_numeric = if (is_continuous) {
          as.numeric(X[[variable]])
        } else {
          NA_real_
        },
        feature_value_category = if (!is_continuous) {
          as.character(X[[variable]])
        } else {
          NA_character_
        },
        shap_value = as.numeric(shap_result$S[, variable]),
        predicted_success_probability = shap_result$probability,
        SHAP_baseline = shap_result$baseline
      )
    }))
  }

  final_consistency_long <- bind_rows(
    final_shap_to_long(
      final_shap_training,
      X_training_final,
      "Training"
    ),
    final_shap_to_long(
      final_shap_internal_test,
      X_internal_test_final,
      "Internal test"
    )
  )

  final_accuracy_audit <- tibble(
    cohort = c("Training", "Internal test"),
    n = c(nrow(X_training_final), nrow(X_internal_test_final)),
    SHAP_baseline = c(
      final_shap_training$baseline,
      final_shap_internal_test$baseline
    ),
    max_local_accuracy_error = c(
      final_shap_training$max_local_accuracy_error,
      final_shap_internal_test$max_local_accuracy_error
    )
  )

  # Quantitative train-test curve concordance using common training quantiles.
  calculate_train_test_concordance <- function(feature_code) {
    feature_data <- final_consistency_long %>%
      filter(.data$feature_code == feature_code)

    if (feature_code %in% continuous_predictors) {
      break_points <- unique(as.numeric(quantile(
        training_data[[feature_code]],
        probs = seq(0, 1, length.out = 11L),
        na.rm = TRUE,
        names = FALSE
      )))
      break_points[1] <- -Inf
      break_points[length(break_points)] <- Inf

      summarized <- feature_data %>%
        mutate(
          comparison_bin = cut(
            feature_value_numeric,
            breaks = break_points,
            include.lowest = TRUE,
            labels = FALSE
          )
        ) %>%
        group_by(cohort, comparison_bin) %>%
        summarise(mean_SHAP = mean(shap_value), .groups = "drop")
    } else {
      summarized <- feature_data %>%
        mutate(comparison_bin = feature_value_category) %>%
        group_by(cohort, comparison_bin) %>%
        summarise(mean_SHAP = mean(shap_value), .groups = "drop")
    }

    summarized %>%
      pivot_wider(names_from = cohort, values_from = mean_SHAP) %>%
      summarise(
        feature_code = feature_code,
        n_common_bins = sum(complete.cases(Training, `Internal test`)),
        curve_Spearman_rho = cor(
          Training,
          `Internal test`,
          method = "spearman",
          use = "complete.obs"
        ),
        curve_RMSE = sqrt(mean(
          (Training - `Internal test`) ^ 2,
          na.rm = TRUE
        ))
      )
  }

  final_curve_concordance <- bind_rows(lapply(
    predictor_names,
    calculate_train_test_concordance
  )) %>%
    mutate(feature = unname(feature_labels[feature_code]))

  cohort_colors <- c(
    "Training" = "#0072B2",
    "Internal test" = "#D55E00"
  )

  make_train_test_continuous_plot <- function(variable) {
    plot_data <- final_consistency_long %>%
      filter(feature_code == variable)

    display_limits <- as.numeric(quantile(
      training_data[[variable]],
      probs = c(0.01, 0.99),
      na.rm = TRUE,
      names = FALSE
    ))

    ggplot(
      plot_data,
      aes(
        x = feature_value_numeric,
        y = shap_value,
        colour = cohort,
        group = cohort
      )
    ) +
      geom_hline(
        yintercept = 0,
        linewidth = 0.4,
        linetype = "dashed",
        colour = "#686868"
      ) +
      geom_point(size = 0.35, alpha = 0.10) +
      geom_smooth(
        method = "loess",
        formula = y ~ x,
        span = 0.80,
        se = FALSE,
        linewidth = 0.78,
        na.rm = TRUE
      ) +
      coord_cartesian(xlim = display_limits) +
      scale_colour_manual(values = cohort_colors) +
      labs(
        title = unname(feature_labels[variable]),
        x = unname(feature_labels_short[variable]),
        y = "SHAP value for P(success)",
        colour = "Cohort"
      ) +
      theme_publication(8.2) +
      theme(legend.position = "bottom")
  }

  make_train_test_categorical_plot <- function(variable) {
    plot_data <- final_consistency_long %>%
      filter(feature_code == variable) %>%
      group_by(cohort, feature_value_category) %>%
      summarise(
        mean_SHAP = mean(shap_value),
        SE = sd(shap_value) / sqrt(n()),
        lower = mean_SHAP - 1.96 * SE,
        upper = mean_SHAP + 1.96 * SE,
        .groups = "drop"
      ) %>%
      mutate(
        feature_value_category = factor(
          feature_value_category,
          levels = stone_location_levels
        )
      )

    dodge <- position_dodge(width = 0.45)

    ggplot(
      plot_data,
      aes(
        x = feature_value_category,
        y = mean_SHAP,
        colour = cohort,
        group = cohort
      )
    ) +
      geom_hline(
        yintercept = 0,
        linewidth = 0.4,
        linetype = "dashed",
        colour = "#686868"
      ) +
      geom_errorbar(
        aes(ymin = lower, ymax = upper),
        position = dodge,
        width = 0.14,
        linewidth = 0.5
      ) +
      geom_point(position = dodge, size = 1.6) +
      scale_colour_manual(values = cohort_colors) +
      scale_x_discrete(labels = c(
        "Distal ureter" = "Distal",
        "Mid ureter" = "Mid",
        "Proximal ureter" = "Proximal"
      )) +
      labs(
        title = unname(feature_labels[variable]),
        x = "Ureteral stone location",
        y = "SHAP value for P(success)",
        colour = "Cohort"
      ) +
      theme_publication(8.2) +
      theme(legend.position = "bottom")
  }

  train_test_plots <- lapply(predictor_names, function(variable) {
    if (variable %in% continuous_predictors) {
      make_train_test_continuous_plot(variable)
    } else {
      make_train_test_categorical_plot(variable)
    }
  })

  figure_train_test <- patchwork::wrap_plots(
    train_test_plots,
    ncol = 2,
    guides = "collect"
  ) +
    patchwork::plot_annotation(
      tag_levels = "A",
      caption = paste0(
        "The locked final XGBoost model was applied without retraining or ",
        "recalibration. The complete training cohort served as the common SHAP ",
        "background distribution for both cohorts. Continuous axes display the ",
        "training-cohort 1st-99th percentile."
      ),
      theme = theme(
        plot.tag = element_text(
          family = base_family,
          face = "bold",
          size = 11
        ),
        plot.caption = element_text(
          family = base_family,
          size = 6.5,
          colour = "#555555",
          hjust = 0
        ),
        legend.position = "bottom"
      )
    )

  write_csv(
    final_accuracy_audit,
    file.path(output_dir, "Final_model_train_test_SHAP_accuracy_audit.csv")
  )
  write_csv(
    final_curve_concordance,
    file.path(output_dir, "Final_model_train_test_SHAP_curve_concordance.csv")
  )
  write_csv(
    final_consistency_long,
    file.path(output_dir, "Final_model_train_test_SHAP_long.csv")
  )
}

# =============================================================================
# 10. Export publication files
# =============================================================================

export_figure <- function(plot_object, stem, width, height) {
  ggsave(
    file.path(output_dir, paste0(stem, ".pdf")),
    plot = plot_object,
    device = grDevices::cairo_pdf,
    width = width,
    height = height,
    units = "in",
    bg = "white"
  )

  ggsave(
    file.path(output_dir, paste0(stem, ".png")),
    plot = plot_object,
    width = width,
    height = height,
    units = "in",
    dpi = 600,
    bg = "white"
  )

  ggsave(
    file.path(output_dir, paste0(stem, ".tiff")),
    plot = plot_object,
    device = "tiff",
    width = width,
    height = height,
    units = "in",
    dpi = 600,
    compression = "lzw",
    bg = "white"
  )
}

export_figure(
  figure_cv_stability,
  "Figure_S2_CV_SHAP_stability_6_variables",
  width = 7.2,
  height = 8.4
)

export_figure(
  figure_cv_summary,
  "Figure_S2b_CV_SHAP_consistency_summary",
  width = 7.2,
  height = 6.2
)

if (isTRUE(run_internal_test_consistency)) {
  export_figure(
    figure_train_test,
    "Figure_SHAP_training_vs_internal_test_consistency_6_variables",
    width = 7.2,
    height = 8.4
  )
}

saveRDS(
  list(
    CV_resamples = cv_resamples,
    CV_SHAP_long = cv_shap_long,
    CV_importance = cv_importance,
    CV_importance_rank_concordance = importance_rank_concordance,
    CV_curve_concordance = cv_curve_concordance,
    CV_local_accuracy_audit = cv_accuracy_audit,
    final_train_test_SHAP_long = final_consistency_long,
    final_train_test_curve_concordance = final_curve_concordance,
    configuration = list(
      CV_seed = cv_seed,
      CV_folds = cv_folds,
      CV_SHAP_nsim = cv_shap_nsim,
      final_model_SHAP_nsim = final_model_shap_nsim,
      probability_target = locked_target_col,
      SHAP_scale = "probability of composite ESWL success"
    )
  ),
  file.path(output_dir, "SHAP_consistency_complete_results.rds")
)

writeLines(
  capture.output(sessionInfo()),
  file.path(output_dir, "SHAP_consistency_sessionInfo.txt")
)

message(
  "SHAP consistency analysis completed.\n",
  "Primary CV figure: Figure_S2_CV_SHAP_stability_6_variables.\n",
  "Output directory: ",
  normalizePath(output_dir, winslash = "/", mustWork = TRUE)
)
