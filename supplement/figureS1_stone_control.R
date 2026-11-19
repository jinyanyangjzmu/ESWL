# ==============================================================================
# Figure S: SFR sensitivity analysis of the locked new XGBoost model
# ==============================================================================
#
# Purpose
#   1. Load the XGBoost workflow already finalized and refitted in the
#      621-patient development cohort.
#   2. Use the model-specific OOF Youden threshold saved during development.
#   3. Replace only the reference outcome with stone-only success (SFR), stored
#      in the `success` column of the three *_sfr cohort objects. In the source
#      RData, this outcome was derived as residual fragment <4 mm = Yes AND
#      repeat lithotripsy = No.
#   4. Evaluate the same locked model in the development, internal-test, and
#      external-validation cohorts without retraining, retuning, recalibration,
#      or threshold reselection.
#   5. Export a publication-quality six-panel figure: three ROC curves above
#      three confusion matrices.
#
# Outcome convention
#   success = 1 is SFR success (the positive reference event).
#   .pred_1 remains the locked model score for the original composite ESWL
#   success outcome. It is not refitted or recalibrated as an SFR probability.
#
# Important interpretation
#   This is an outcome-definition sensitivity analysis of the original locked
#   model, not development of a new SFR model. Performance obtained by applying
#   the final refitted model to the same 621 development cases is apparent.
# ==============================================================================


# ==============================================================================
# 0. User settings: edit paths here only
# ==============================================================================

data_file <- paste0(
  "data/",
  "单纯碎石成功的三队列.RDATA"
)

model_dir <- paste0(
  "output/",
  "model_benchmark_9_CV_FORMAL/13_final_models"
)

xgb_model_file <- file.path(
  model_dir,
  "04_XGBoost_best_CV_refit_621.rds"
)

all_models_bundle_file <- file.path(
  model_dir,
  "00_ESWL_all_9_best_CV_refit_models.rds"
)

output_dir <- paste0(
  "output/",
  "figureS_SFR_sensitivity_new_xgboost_three_cohorts"
)

BASE_FAMILY <- "Arial"
FIGURE_WIDTH_IN <- 11.8
FIGURE_HEIGHT_IN <- 6.7
RASTER_DPI <- 600L

EXPECTED_COHORT_N <- c(
  "Development cohort" = 621L,
  "Internal test cohort" = 208L,
  "External validation cohort" = 147L
)

# These event counts were independently audited from the exact uploaded RData
# file (MD5: 31f78ad736d0db766269b745b39f93b2). They prevent accidental use of an
# older three-cohort file that happens to contain objects with the same names.
EXPECTED_SFR_SUCCESS_N <- c(
  "Development cohort" = 539L,
  "Internal test cohort" = 180L,
  "External validation cohort" = 136L
)

EXPECTED_SFR_FAILURE_N <- c(
  "Development cohort" = 82L,
  "Internal test cohort" = 28L,
  "External validation cohort" = 11L
)


# ==============================================================================
# 1. Packages and output directory
# ==============================================================================

required_packages <- c(
  "tidymodels", "dplyr", "tidyr", "purrr", "tibble", "readr",
  "ggplot2", "pROC", "patchwork", "scales", "xgboost"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0L) {
  stop(
    "Install missing packages first: ",
    paste(missing_packages, collapse = ", ")
  )
}

suppressPackageStartupMessages({
  library(tidymodels)
})

if (!dir.exists(output_dir) &&
    !dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)) {
  stop("Cannot create output directory: ", output_dir)
}

for (required_file in c(data_file, xgb_model_file, all_models_bundle_file)) {
  if (!file.exists(required_file)) {
    stop("Required file does not exist: ", required_file)
  }
}


# ==============================================================================
# 2. Load the three cohorts without polluting the global environment
# ==============================================================================

data_environment <- new.env(parent = emptyenv())
loaded_object_names <- load(data_file, envir = data_environment)

get_cohort_object <- function(preferred_names, cohort_label) {
  available_names <- preferred_names[
    preferred_names %in% loaded_object_names
  ]

  if (length(available_names) == 0L) {
    stop(
      "Could not find the ", cohort_label, " object in ", basename(data_file),
      ". Expected one of: ", paste(preferred_names, collapse = ", "),
      ". Available objects: ", paste(loaded_object_names, collapse = ", ")
    )
  }

  selected_name <- available_names[[1L]]
  message(cohort_label, " object: ", selected_name)
  object <- get(selected_name, envir = data_environment, inherits = FALSE)

  if (!is.data.frame(object)) {
    stop(cohort_label, " object is not a data frame/tibble.")
  }
  object
}

training_raw <- get_cohort_object(
  "training_data_sfr",
  "Development cohort"
)

internal_test_raw <- get_cohort_object(
  "test_data_sfr",
  "Internal test cohort"
)

external_raw <- get_cohort_object(
  "external_data_sfr",
  "External validation cohort"
)


# ==============================================================================
# 3. Strictly harmonize names, types, units, SFR outcome, and stone location
# ==============================================================================

resolve_column <- function(data, candidate_names, variable_label, cohort_label) {
  matched_names <- candidate_names[candidate_names %in% names(data)]

  if (length(matched_names) == 0L) {
    stop(
      cohort_label, " is missing ", variable_label, ". Expected one of: ",
      paste(candidate_names, collapse = ", ")
    )
  }

  if (length(matched_names) > 1L) {
    reference_value <- data[[matched_names[[1L]]]]
    equivalent <- vapply(
      matched_names[-1L],
      function(column_name) {
        isTRUE(all.equal(
          as.character(reference_value),
          as.character(data[[column_name]]),
          check.attributes = FALSE
        ))
      },
      logical(1)
    )

    if (!all(equivalent)) {
      stop(
        cohort_label, " contains conflicting columns for ", variable_label,
        ": ", paste(matched_names, collapse = ", ")
      )
    }
  }

  data[[matched_names[[1L]]]]
}

strict_numeric <- function(x, variable_label, cohort_label) {
  raw <- trimws(as.character(x))
  raw[is.na(x) | raw == ""] <- NA_character_
  converted <- suppressWarnings(as.numeric(raw))

  invalid <- is.na(converted) | !is.finite(converted)
  if (any(invalid)) {
    stop(
      cohort_label, ": ", variable_label,
      " contains missing/non-numeric/non-finite values: ",
      paste(utils::head(unique(raw[invalid]), 10L), collapse = ", ")
    )
  }
  converted
}

parse_success <- function(x, cohort_label) {
  raw <- trimws(tolower(as.character(x)))

  parsed <- dplyr::case_when(
    raw %in% c("yes", "1", "success", "successful") ~ "1",
    raw %in% c("no", "0", "failure", "failed", "unsuccessful") ~ "0",
    TRUE ~ NA_character_
  )

  if (anyNA(parsed)) {
    stop(
      cohort_label, ": success contains missing/unrecognized values: ",
      paste(utils::head(unique(raw[is.na(parsed)]), 10L), collapse = ", ")
    )
  }

  factor(parsed, levels = c("1", "0"))
}

parse_stone_location <- function(x, cohort_label) {
  raw <- trimws(gsub("[[:space:]]+", " ", as.character(x)))
  key <- tolower(raw)

  parsed <- dplyr::case_when(
    key %in% c("proximal ureter", "proximal") ~ "Proximal ureter",
    key %in% c("mid ureter", "middle ureter", "mid", "middle") ~
      "Mid ureter",
    key %in% c("distal ureter", "distal") ~ "Distal ureter",
    TRUE ~ NA_character_
  )

  if (anyNA(parsed)) {
    stop(
      cohort_label, ": Stone location contains missing/unrecognized levels: ",
      paste(utils::head(unique(raw[is.na(parsed)]), 10L), collapse = ", ")
    )
  }

  factor(
    parsed,
    levels = c("Distal ureter", "Mid ureter", "Proximal ureter")
  )
}

prepare_cohort <- function(data, cohort_label) {
  outcome <- resolve_column(
    data,
    "success",
    "SFR outcome column `success`",
    cohort_label
  )

  prepared <- tibble::tibble(
    Case_ID = seq_len(nrow(data)),
    success = parse_success(outcome, cohort_label),
    Stone_volume = strict_numeric(
      resolve_column(
        data,
        c("Stone_volume", "Stone volume", "Stone.volume"),
        "Stone volume",
        cohort_label
      ),
      "Stone volume",
      cohort_label
    ),
    Mean_CT_attenuation = strict_numeric(
      resolve_column(
        data,
        c(
          "Mean_CT_attenuation", "Mean CT attenuation",
          "Mean.CT.attenuation"
        ),
        "Mean CT attenuation",
        cohort_label
      ),
      "Mean CT attenuation",
      cohort_label
    ),
    Ureteral_wall_thickness = strict_numeric(
      resolve_column(
        data,
        c(
          "Ureteral_wall_thickness", "Ureteral wall thickness",
          "Ureteral.wall.thickness"
        ),
        "Ureteral wall thickness",
        cohort_label
      ),
      "Ureteral wall thickness",
      cohort_label
    ),
    Hydronephrosis_measure = strict_numeric(
      resolve_column(
        data,
        c(
          "Hydronephrosis_measure", "Degree of hydronephrosis",
          "Degree.of.hydronephrosis", "hydronephrosis", "Hydronephrosis"
        ),
        "continuous hydronephrosis measure",
        cohort_label
      ),
      "continuous hydronephrosis measure",
      cohort_label
    ),
    Subcutaneous_fat_thickness = strict_numeric(
      resolve_column(
        data,
        c(
          "Subcutaneous_fat_thickness", "Subcutaneous fat thickness",
          "Subcutaneous.fat.thickness"
        ),
        "Subcutaneous fat thickness",
        cohort_label
      ),
      "Subcutaneous fat thickness",
      cohort_label
    ),
    Stone_location = parse_stone_location(
      resolve_column(
        data,
        c("Stone_location", "Stone location", "Stone.location"),
        "Stone location",
        cohort_label
      ),
      cohort_label
    )
  )

  if (nrow(prepared) == 0L || dplyr::n_distinct(prepared$success) != 2L) {
    stop(cohort_label, " must contain both success and failure outcomes.")
  }

  prepared
}

cohort_data <- list(
  "Development cohort" = prepare_cohort(training_raw, "Development cohort"),
  "Internal test cohort" = prepare_cohort(
    internal_test_raw,
    "Internal test cohort"
  ),
  "External validation cohort" = prepare_cohort(
    external_raw,
    "External validation cohort"
  )
)

observed_cohort_n <- vapply(cohort_data, nrow, integer(1))
if (!identical(observed_cohort_n, EXPECTED_COHORT_N[names(cohort_data)])) {
  stop(
    "Unexpected cohort sizes. Observed: ",
    paste(names(observed_cohort_n), observed_cohort_n, sep = "=", collapse = "; "),
    ". Expected: ",
    paste(
      names(EXPECTED_COHORT_N),
      EXPECTED_COHORT_N,
      sep = "=",
      collapse = "; "
    )
  )
}

observed_success_n <- vapply(
  cohort_data,
  function(data) sum(data$success == "1"),
  integer(1)
)
observed_failure_n <- vapply(
  cohort_data,
  function(data) sum(data$success == "0"),
  integer(1)
)

if (!identical(
  observed_success_n,
  EXPECTED_SFR_SUCCESS_N[names(cohort_data)]
) || !identical(
  observed_failure_n,
  EXPECTED_SFR_FAILURE_N[names(cohort_data)]
)) {
  stop(
    "The SFR event counts do not match the audited RData. Observed success: ",
    paste(
      names(observed_success_n), observed_success_n,
      sep = "=", collapse = "; "
    ),
    ". Observed failure: ",
    paste(
      names(observed_failure_n), observed_failure_n,
      sep = "=", collapse = "; "
    ),
    ". Expected success/failure: 539/82, 180/28, and 136/11."
  )
}

# Recompute the alternative outcome from the raw component columns retained in
# this exact RData. This audit is intentionally tied to the supplied file.
audit_sfr_definition <- function(raw_follow_up, saved_outcome, cohort_label) {
  residual_name <- "Residual fragment maximal diameter <4 mm (Yes/No)"
  repeat_name <- "Repeat lithotripsy(Yes/No)"

  if (!all(c(residual_name, repeat_name) %in% names(raw_follow_up))) {
    stop(cohort_label, ": source columns for the SFR definition are missing.")
  }

  residual <- trimws(as.character(raw_follow_up[[residual_name]]))
  repeat_treatment <- trimws(as.character(raw_follow_up[[repeat_name]]))

  if (anyNA(residual) || anyNA(repeat_treatment) ||
      any(!residual %in% c("Yes", "No")) ||
      any(!repeat_treatment %in% c("Yes", "No"))) {
    stop(cohort_label, ": invalid source values in the SFR definition.")
  }

  recomputed <- ifelse(
    residual == "Yes" & repeat_treatment == "No",
    "Yes",
    "No"
  )

  if (!identical(recomputed, trimws(as.character(saved_outcome)))) {
    stop(
      cohort_label,
      ": saved `success` does not match residual fragment <4 mm = Yes AND ",
      "repeat lithotripsy = No."
    )
  }
}

if (all(c("recover", "ESWL1") %in% loaded_object_names)) {
  audit_sfr_definition(
    get("recover", envir = data_environment, inherits = FALSE),
    get("ESWL1", envir = data_environment, inherits = FALSE)$success,
    "Development source data"
  )
}

if (all(c("external_full", "external_data_sfr") %in% loaded_object_names)) {
  audit_sfr_definition(
    get("external_full", envir = data_environment, inherits = FALSE),
    get("external_data_sfr", envir = data_environment, inherits = FALSE)$success,
    "External-validation source data"
  )
}

rm(training_raw, internal_test_raw, external_raw, data_environment)


# ==============================================================================
# 4. Load and lock the new XGBoost workflow and its OOF threshold
# ==============================================================================

xgb_model <- readRDS(xgb_model_file)
model_bundle <- readRDS(all_models_bundle_file)

if (!is.list(model_bundle) ||
    !all(c("models", "thresholds") %in% names(model_bundle))) {
  stop("The all-model bundle does not contain models and thresholds.")
}

if (!"XGBoost" %in% names(model_bundle$models)) {
  stop("The all-model bundle does not contain an XGBoost model.")
}

threshold_rows <- model_bundle$thresholds |>
  dplyr::filter(Model == "XGBoost")

if (nrow(threshold_rows) != 1L ||
    !"OOF_Youden_threshold" %in% names(threshold_rows)) {
  stop("Could not identify one XGBoost OOF Youden threshold.")
}

fixed_threshold <- as.numeric(threshold_rows$OOF_Youden_threshold[[1L]])

if (length(fixed_threshold) != 1L ||
    !is.finite(fixed_threshold) ||
    fixed_threshold <= 0 || fixed_threshold >= 1) {
  stop("The saved XGBoost OOF threshold is invalid.")
}

# Confirm that the separately saved XGBoost workflow and the bundle copy are
# exactly prediction-equivalent before evaluating any cohort.
model_check_data <- dplyr::slice_head(cohort_data[[1L]], n = 10L)
individual_check <- predict(
  xgb_model,
  new_data = model_check_data,
  type = "prob"
)$.pred_1
bundle_check <- predict(
  model_bundle$models$XGBoost,
  new_data = model_check_data,
  type = "prob"
)$.pred_1

if (!isTRUE(all.equal(
  individual_check,
  bundle_check,
  tolerance = 1e-12,
  check.attributes = FALSE
))) {
  stop(
    "The individual XGBoost file and bundle XGBoost model are not equivalent."
  )
}

rm(individual_check, bundle_check, model_check_data, model_bundle)


# ==============================================================================
# 5. Generate locked probabilities in all three cohorts
# ==============================================================================

cohort_order <- names(cohort_data)

prediction_data <- purrr::imap_dfr(
  cohort_data,
  function(data, cohort_name) {
    probability <- predict(
      xgb_model,
      new_data = data,
      type = "prob"
    )

    if (!".pred_1" %in% names(probability) ||
        nrow(probability) != nrow(data) ||
        any(!is.finite(probability$.pred_1)) ||
        any(probability$.pred_1 < 0 | probability$.pred_1 > 1)) {
      stop(cohort_name, ": invalid .pred_1 probabilities.")
    }

    tibble::tibble(
      Cohort = cohort_name,
      Case_ID = data$Case_ID,
      success = data$success,
      truth = as.integer(data$success == "1"),
      .pred_1 = probability$.pred_1,
      Fixed_OOF_threshold = fixed_threshold,
      predicted_class = ifelse(
        probability$.pred_1 >= fixed_threshold,
        1L,
        0L
      )
    )
  }
) |>
  dplyr::mutate(
    Cohort = factor(Cohort, levels = cohort_order)
  )


# ==============================================================================
# 6. AUC, DeLong CI, confusion matrices, and exact binomial CIs
# ==============================================================================

exact_binomial <- function(numerator, denominator) {
  if (denominator == 0L) {
    return(c(estimate = NA_real_, lower = NA_real_, upper = NA_real_))
  }

  interval <- stats::binom.test(numerator, denominator)$conf.int
  c(
    estimate = numerator / denominator,
    lower = unname(interval[[1L]]),
    upper = unname(interval[[2L]])
  )
}

evaluate_one_cohort <- function(data, cohort_name) {
  roc_object <- pROC::roc(
    response = data$truth,
    predictor = data$.pred_1,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )

  auc_value <- as.numeric(pROC::auc(roc_object))
  auc_ci <- as.numeric(pROC::ci.auc(
    roc_object,
    conf.level = 0.95,
    method = "delong"
  ))

  if (!is.finite(auc_value)) {
    stop(cohort_name, ": AUC could not be calculated.")
  }

  if (auc_value < 0.5) {
    warning(
      cohort_name,
      ": AUC is below 0.5. The calculation is retained, but verify the SFR ",
      "outcome coding and remember that .pred_1 targets the original ",
      "composite outcome.",
      call. = FALSE
    )
  }

  tn <- sum(data$truth == 0L & data$predicted_class == 0L)
  fp <- sum(data$truth == 0L & data$predicted_class == 1L)
  fn <- sum(data$truth == 1L & data$predicted_class == 0L)
  tp <- sum(data$truth == 1L & data$predicted_class == 1L)

  accuracy <- exact_binomial(tp + tn, tp + tn + fp + fn)
  sensitivity <- exact_binomial(tp, tp + fn)
  specificity <- exact_binomial(tn, tn + fp)
  ppv <- exact_binomial(tp, tp + fp)
  npv <- exact_binomial(tn, tn + fn)

  roc_coordinates <- tibble::tibble(
    False_positive_rate = 1 - roc_object$specificities,
    Sensitivity = roc_object$sensitivities,
    Cohort = cohort_name
  ) |>
    dplyr::distinct(False_positive_rate, Sensitivity, .keep_all = TRUE) |>
    dplyr::arrange(False_positive_rate, Sensitivity)

  confusion <- tidyr::expand_grid(
    Prediction = c("0", "1"),
    Truth = c("0", "1")
  ) |>
    dplyr::mutate(
      Count = dplyr::case_when(
        Prediction == "0" & Truth == "0" ~ tn,
        Prediction == "1" & Truth == "0" ~ fp,
        Prediction == "0" & Truth == "1" ~ fn,
        Prediction == "1" & Truth == "1" ~ tp
      ),
      Cell_proportion = Count / nrow(data),
      Cell_label = sprintf("%d\n(%.1f%%)", Count, 100 * Cell_proportion),
      Cohort = cohort_name
    )

  summary <- tibble::tibble(
    Cohort = cohort_name,
    N = nrow(data),
    Success_n = sum(data$truth == 1L),
    Failure_n = sum(data$truth == 0L),
    Fixed_OOF_threshold = fixed_threshold,
    AUC = auc_value,
    AUC_lower = auc_ci[[1L]],
    AUC_upper = auc_ci[[3L]],
    Accuracy = accuracy[["estimate"]],
    Accuracy_lower = accuracy[["lower"]],
    Accuracy_upper = accuracy[["upper"]],
    Sensitivity = sensitivity[["estimate"]],
    Sensitivity_lower = sensitivity[["lower"]],
    Sensitivity_upper = sensitivity[["upper"]],
    Specificity = specificity[["estimate"]],
    Specificity_lower = specificity[["lower"]],
    Specificity_upper = specificity[["upper"]],
    PPV = ppv[["estimate"]],
    PPV_lower = ppv[["lower"]],
    PPV_upper = ppv[["upper"]],
    NPV = npv[["estimate"]],
    NPV_lower = npv[["lower"]],
    NPV_upper = npv[["upper"]],
    TN = tn,
    FP = fp,
    FN = fn,
    TP = tp
  )

  list(
    roc_object = roc_object,
    roc_coordinates = roc_coordinates,
    confusion = confusion,
    summary = summary
  )
}

evaluation_results <- lapply(
  cohort_order,
  function(cohort_name) {
    cohort_predictions <- prediction_data |>
      dplyr::filter(Cohort == cohort_name)
    evaluate_one_cohort(cohort_predictions, cohort_name)
  }
)
names(evaluation_results) <- cohort_order

roc_data <- purrr::map_dfr(evaluation_results, "roc_coordinates")
confusion_data <- purrr::map_dfr(evaluation_results, "confusion")
performance_raw <- purrr::map_dfr(evaluation_results, "summary") |>
  dplyr::mutate(Cohort = factor(Cohort, levels = cohort_order)) |>
  dplyr::arrange(Cohort)

format_estimate_ci <- function(estimate, lower, upper) {
  ifelse(
    is.finite(estimate) & is.finite(lower) & is.finite(upper),
    sprintf("%.3f (%.3f-%.3f)", estimate, lower, upper),
    NA_character_
  )
}

performance_formatted <- performance_raw |>
  dplyr::transmute(
    Cohort = as.character(Cohort),
    N,
    Success_n,
    Failure_n,
    Fixed_OOF_threshold = sprintf("%.6f", Fixed_OOF_threshold),
    `AUC (95% CI)` = format_estimate_ci(AUC, AUC_lower, AUC_upper),
    `Accuracy (95% CI)` = format_estimate_ci(
      Accuracy, Accuracy_lower, Accuracy_upper
    ),
    `Sensitivity (95% CI)` = format_estimate_ci(
      Sensitivity, Sensitivity_lower, Sensitivity_upper
    ),
    `Specificity (95% CI)` = format_estimate_ci(
      Specificity, Specificity_lower, Specificity_upper
    ),
    `PPV (95% CI)` = format_estimate_ci(PPV, PPV_lower, PPV_upper),
    `NPV (95% CI)` = format_estimate_ci(NPV, NPV_lower, NPV_upper),
    TN,
    FP,
    FN,
    TP
  )


# ==============================================================================
# 7. Publication-quality sensitivity-analysis figure
# ==============================================================================

cohort_display <- c(
  "Development cohort" = "Training Set (Apparent)",
  "Internal test cohort" = "Internal Test Set",
  "External validation cohort" = "External Validation Set"
)

roc_letters <- c("A", "B", "C")
cm_letters <- c("D", "E", "F")

base_theme <- ggplot2::theme_bw(
  base_size = 9,
  base_family = BASE_FAMILY
) +
  ggplot2::theme(
    plot.title = ggplot2::element_text(
      size = 10,
      face = "bold",
      hjust = 0,
      margin = ggplot2::margin(b = 2)
    ),
    plot.subtitle = ggplot2::element_text(
      size = 8.4,
      color = "#333333",
      hjust = 0,
      margin = ggplot2::margin(b = 5)
    ),
    axis.title = ggplot2::element_text(size = 9, color = "#222222"),
    axis.text = ggplot2::element_text(size = 8, color = "#222222"),
    panel.grid.major = ggplot2::element_line(
      color = "#D9DDE5",
      linewidth = 0.45
    ),
    panel.grid.minor = ggplot2::element_blank(),
    panel.border = ggplot2::element_rect(
      color = "#333333",
      linewidth = 0.45
    ),
    plot.margin = ggplot2::margin(7, 8, 7, 7)
  )

make_roc_plot <- function(cohort_name, panel_letter) {
  plot_data <- roc_data |>
    dplyr::filter(Cohort == cohort_name)

  summary_row <- performance_raw |>
    dplyr::filter(Cohort == cohort_name)

  subtitle_text <- sprintf(
    "AUC = %.3f (95%% CI %.3f-%.3f); n = %d",
    summary_row$AUC,
    summary_row$AUC_lower,
    summary_row$AUC_upper,
    summary_row$N
  )

  ggplot2::ggplot(
    plot_data,
    ggplot2::aes(x = False_positive_rate, y = Sensitivity)
  ) +
    ggplot2::geom_abline(
      intercept = 0,
      slope = 1,
      color = "#D97989",
      linewidth = 0.7,
      linetype = "dashed"
    ) +
    ggplot2::geom_step(
      color = "#5046A5",
      linewidth = 0.95,
      direction = "vh"
    ) +
    ggplot2::scale_x_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, by = 0.25),
      labels = scales::label_number(accuracy = 0.01),
      expand = c(0, 0)
    ) +
    ggplot2::scale_y_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, by = 0.25),
      labels = scales::label_number(accuracy = 0.01),
      expand = c(0, 0)
    ) +
    ggplot2::coord_equal() +
    ggplot2::labs(
      title = paste(
        panel_letter,
        " ROC Curve for ",
        cohort_display[[cohort_name]],
        sep = ""
      ),
      subtitle = subtitle_text,
      x = "1 - Specificity",
      y = "Sensitivity"
    ) +
    base_theme
}

maximum_cell_proportion <- max(confusion_data$Cell_proportion)

make_confusion_plot <- function(cohort_name, panel_letter) {
  plot_data <- confusion_data |>
    dplyr::filter(Cohort == cohort_name) |>
    dplyr::mutate(
      Text_colour = ifelse(Cell_proportion >= 0.35, "white", "#17202A")
    )

  cohort_n <- performance_raw |>
    dplyr::filter(Cohort == cohort_name) |>
    dplyr::pull(N)

  ggplot2::ggplot(
    plot_data,
    ggplot2::aes(x = Truth, y = Prediction, fill = Cell_proportion)
  ) +
    ggplot2::geom_tile(color = "white", linewidth = 1.0) +
    ggplot2::geom_text(
      ggplot2::aes(label = Cell_label, color = Text_colour),
      size = 3.1,
      lineheight = 0.92,
      family = BASE_FAMILY,
      fontface = "bold"
    ) +
    ggplot2::scale_color_identity() +
    ggplot2::scale_fill_gradient(
      low = "#F1F7FA",
      high = "#2E78A6",
      limits = c(0, maximum_cell_proportion),
      oob = scales::squish,
      guide = "none"
    ) +
    ggplot2::scale_x_discrete(drop = FALSE) +
    ggplot2::scale_y_discrete(limits = c("1", "0"), drop = FALSE) +
    ggplot2::coord_fixed() +
    ggplot2::labs(
      title = paste(
        panel_letter,
        " Confusion Matrix for ",
        cohort_display[[cohort_name]],
        sep = ""
      ),
      subtitle = sprintf(
        "Fixed development OOF threshold = %.3f; n = %d",
        fixed_threshold,
        cohort_n
      ),
      x = "Observed stone-only success",
      y = "Locked prediction"
    ) +
    base_theme +
    ggplot2::theme(
      panel.grid = ggplot2::element_blank(),
      axis.ticks = ggplot2::element_blank()
    )
}

roc_plots <- purrr::map2(
  cohort_order,
  roc_letters,
  make_roc_plot
)

confusion_plots <- purrr::map2(
  cohort_order,
  cm_letters,
  make_confusion_plot
)

figure_sfr <- patchwork::wrap_plots(
  plotlist = c(roc_plots, confusion_plots),
  ncol = 3,
  nrow = 2,
  widths = c(1, 1, 1),
  heights = c(1.10, 0.90)
)


# ==============================================================================
# 8. Export figure, predictions, metrics, and reproducibility metadata
# ==============================================================================

pdf_file <- file.path(
  output_dir,
  "FigureS_SFR_sensitivity_XGBoost_three_cohorts.pdf"
)
tiff_file <- file.path(
  output_dir,
  "FigureS_SFR_sensitivity_XGBoost_three_cohorts_600dpi.tiff"
)
png_file <- file.path(
  output_dir,
  "FigureS_SFR_sensitivity_XGBoost_three_cohorts_600dpi.png"
)

if (capabilities("cairo")) {
  ggplot2::ggsave(
    filename = pdf_file,
    plot = figure_sfr,
    device = grDevices::cairo_pdf,
    width = FIGURE_WIDTH_IN,
    height = FIGURE_HEIGHT_IN,
    units = "in",
    bg = "white",
    limitsize = FALSE
  )
} else {
  ggplot2::ggsave(
    filename = pdf_file,
    plot = figure_sfr,
    device = "pdf",
    width = FIGURE_WIDTH_IN,
    height = FIGURE_HEIGHT_IN,
    units = "in",
    bg = "white",
    useDingbats = FALSE,
    limitsize = FALSE
  )
}

ggplot2::ggsave(
  filename = tiff_file,
  plot = figure_sfr,
  device = "tiff",
  width = FIGURE_WIDTH_IN,
  height = FIGURE_HEIGHT_IN,
  units = "in",
  dpi = RASTER_DPI,
  compression = "lzw",
  bg = "white",
  limitsize = FALSE
)

ggplot2::ggsave(
  filename = png_file,
  plot = figure_sfr,
  device = "png",
  width = FIGURE_WIDTH_IN,
  height = FIGURE_HEIGHT_IN,
  units = "in",
  dpi = RASTER_DPI,
  bg = "white",
  limitsize = FALSE
)

readr::write_csv(
  prediction_data |>
    dplyr::mutate(
      Cohort = as.character(Cohort),
      success = as.character(success)
    ),
  file.path(output_dir, "01_SFR_sensitivity_predictions.csv")
)

readr::write_csv(
  performance_raw |>
    dplyr::mutate(Cohort = as.character(Cohort)),
  file.path(output_dir, "02_SFR_sensitivity_performance_raw.csv")
)

readr::write_csv(
  performance_formatted,
  file.path(output_dir, "03_SFR_sensitivity_performance_formatted.csv")
)

readr::write_csv(
  confusion_data,
  file.path(output_dir, "04_SFR_sensitivity_confusion_cells.csv")
)

saveRDS(
  list(
    model_file = basename(xgb_model_file),
    model_file_md5 = unname(tools::md5sum(xgb_model_file)),
    model_bundle_file = basename(all_models_bundle_file),
    model_bundle_md5 = unname(tools::md5sum(all_models_bundle_file)),
    data_file = basename(data_file),
    data_file_md5 = unname(tools::md5sum(data_file)),
    analysis_type = "Outcome-definition sensitivity analysis",
    reference_outcome = paste0(
      "Stone-only success: residual fragment <4 mm = Yes AND ",
      "repeat lithotripsy = No"
    ),
    source_outcome_column = "success",
    audited_success_counts = observed_success_n,
    audited_failure_counts = observed_failure_n,
    positive_event = "1",
    model_score_target = "Original composite ESWL success",
    probability_column = ".pred_1",
    model_retrained = FALSE,
    model_recalibrated = FALSE,
    threshold_reselected = FALSE,
    fixed_OOF_Youden_threshold = fixed_threshold,
    predictions = prediction_data,
    performance_raw = performance_raw,
    performance_formatted = performance_formatted,
    confusion_cells = confusion_data
  ),
  file.path(output_dir, "05_SFR_sensitivity_evaluation_results.rds")
)

writeLines(
  capture.output(utils::sessionInfo()),
  file.path(output_dir, "06_sessionInfo.txt")
)

print(performance_formatted)
print(figure_sfr)

message("Completed SFR sensitivity analysis without retraining, retuning, ",
        "recalibration, or threshold reselection.")
message("Fixed XGBoost OOF Youden threshold = ", signif(fixed_threshold, 8L))
message("Outputs saved to: ", output_dir)
