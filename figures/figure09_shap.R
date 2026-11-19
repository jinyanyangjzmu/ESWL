# =============================================================================
# Publication-grade SHAP analysis for the locked XGBoost model
# Outcome: composite first-session ESWL success (Yes)
# Cohort used for the primary global explanation: development/training cohort
# =============================================================================

# Install once if needed:
# install.packages(c(
#   "tidymodels", "dplyr", "tidyr", "readr", "fastshap", "shapviz",
#   "ggplot2", "patchwork", "scales", "ggbeeswarm"
# ))

suppressPackageStartupMessages({
  library(tidymodels)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(fastshap)
  library(shapviz)
  library(ggplot2)
  library(patchwork)
  library(scales)
  library(ggbeeswarm)
})
# =============================================================================
# 1. Configuration: edit only this section
# =============================================================================

# Put this script, the model RDS, and the training-data RDS in the same folder,
# or replace the two paths below with absolute paths.
model_file <- 'output/model_benchmark_9_CV_FORMAL/13_final_models/04_XGBoost_best_CV_refit_621.rds'
train_file <-'output/feature_selection/01_traindata.rds'
output_dir <- "output/SHAP_composite_success_new_XGBoost"

set.seed(20260904)

# Number of Monte Carlo repetitions for probability-scale FastSHAP.
# 500 is suitable for the final run; use 50 only for a quick code check.
shap_nsim <- 500L

# Pre-specified representative patients for individual explanations.
representative_quantiles <- c(lower = 0.10, median = 0.50, higher = 0.90)

if (!file.exists(model_file)) {
  stop("Model file not found: ", normalizePath(model_file, mustWork = FALSE))
}
if (!file.exists(train_file)) {
  stop("Training-data file not found: ", normalizePath(train_file, mustWork = FALSE))
}
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Use Arial on Windows and a portable sans-serif fallback elsewhere.
base_family <- if (.Platform$OS.type == "windows") "Arial" else "sans"

# =============================================================================
# 2. Read the locked model and reconstruct its exact raw predictor interface
# =============================================================================

locked_xgboost <- readRDS(model_file)
training_raw <- readRDS(train_file)

required_source_columns <- c(
  "Stone volume",
  "Mean CT attenuation",
  "Ureteral wall thickness",
  "Degree of hydronephrosis",
  "Subcutaneous fat thickness",
  "Stone location",
  "success"
)

missing_source_columns <- setdiff(required_source_columns, names(training_raw))
if (length(missing_source_columns) > 0L) {
  stop(
    "The training RDS is missing required columns: ",
    paste(missing_source_columns, collapse = ", ")
  )
}

# These names and factor levels match the fitted workflow stored in the RDS.
stone_location_levels <- c("Distal ureter", "Mid ureter", "Proximal ureter")

training_model <- training_raw %>%
  transmute(
    Case_ID = dplyr::row_number(),
    Stone_volume = as.numeric(.data[["Stone volume"]]),
    Mean_CT_attenuation = as.numeric(.data[["Mean CT attenuation"]]),
    Ureteral_wall_thickness = as.numeric(.data[["Ureteral wall thickness"]]),
    Hydronephrosis_measure = as.numeric(.data[["Degree of hydronephrosis"]]),
    Subcutaneous_fat_thickness = as.numeric(.data[["Subcutaneous fat thickness"]]),
    Stone_location = factor(
      as.character(.data[["Stone location"]]),
      levels = stone_location_levels
    ),
    success = factor(as.character(.data[["success"]]), levels = c("No", "Yes"))
  )

if (anyNA(training_model)) {
  na_counts <- vapply(training_model, function(z) sum(is.na(z)), integer(1))
  stop(
    "Missing or unrecognized values were found after data reconstruction: ",
    paste(names(na_counts)[na_counts > 0L], na_counts[na_counts > 0L],
          sep = "=", collapse = ", ")
  )
}

if (!identical(levels(training_model$success), c("No", "Yes"))) {
  stop("Outcome factor levels are not c('No', 'Yes').")
}

predictor_names <- c(
  "Stone_volume",
  "Mean_CT_attenuation",
  "Ureteral_wall_thickness",
  "Hydronephrosis_measure",
  "Subcutaneous_fat_thickness",
  "Stone_location"
)

X <- training_model %>%
  select(all_of(predictor_names)) %>%
  as.data.frame()

# Probe the locked workflow before SHAP calculation. Depending on how the
# outcome was coded when the workflow was fitted, the success-probability
# column can be either .pred_1 (0/1 outcome) or .pred_Yes (No/Yes outcome).
probe_prediction <- predict(
  locked_xgboost,
  new_data = training_model,
  type = "prob"
)

success_probability_candidates <- c(".pred_1", ".pred_Yes")
available_success_columns <- intersect(
  success_probability_candidates,
  names(probe_prediction)
)

if (length(available_success_columns) != 1L) {
  stop(
    "Could not identify exactly one success-probability column. Available columns: ",
    paste(names(probe_prediction), collapse = ", "),
    ". Expected either .pred_1 or .pred_Yes."
  )
}

target_col <- available_success_columns[[1L]]
message("Success-probability column identified as: ", target_col)

# The workflow was trained with Case_ID assigned a non-predictor role. Add a
# temporary ID during FastSHAP permutations so the workflow receives the same
# raw-data interface; this ID never enters the XGBoost predictor matrix.
prepare_prediction_data <- function(newdata) {
  newdata <- as.data.frame(newdata)

  missing_predictors <- setdiff(predictor_names, names(newdata))
  if (length(missing_predictors) > 0L) {
    stop("Prediction data are missing: ", paste(missing_predictors, collapse = ", "))
  }

  newdata$Stone_location <- factor(
    as.character(newdata$Stone_location),
    levels = stone_location_levels
  )

  if (anyNA(newdata$Stone_location)) {
    stop("An unrecognized Stone_location level was generated during prediction.")
  }

  newdata$Case_ID <- seq_len(nrow(newdata))
  newdata %>% select(Case_ID, all_of(predictor_names))
}

pred_success <- function(object, newdata) {
  prediction_data <- prepare_prediction_data(newdata)
  prediction <- predict(object, new_data = prediction_data, type = "prob")

  if (!target_col %in% names(prediction)) {
    stop("Probability column ", target_col, " was not returned by the model.")
  }
  as.numeric(prediction[[target_col]])
}

p_success <- pred_success(locked_xgboost, X)

if (length(p_success) != nrow(X) || any(!is.finite(p_success)) ||
    any(p_success < 0 | p_success > 1)) {
  stop("Invalid success probabilities were returned by the locked model.")
}

# Direction audit against the original clinical outcome. This does not choose
# or recalibrate the model; it only prevents accidental use of failure
# probability. The predicted success probability should be higher, on average,
# in clinically successful patients than in unsuccessful patients.
mean_probability_by_outcome <- tapply(
  p_success,
  training_model$success,
  mean,
  na.rm = TRUE
)

if (!all(c("No", "Yes") %in% names(mean_probability_by_outcome))) {
  stop("Both observed outcome groups ('No' and 'Yes') are required for direction checking.")
}

if (unname(mean_probability_by_outcome["Yes"]) <=
    unname(mean_probability_by_outcome["No"])) {
  stop(
    "Probability-direction check failed: the selected column ", target_col,
    " is not higher on average among observed successes. Do not continue until ",
    "the outcome coding of the fitted workflow has been verified."
  )
}

message(sprintf(
  paste0(
    "Direction check passed: mean predicted P(success) = %.3f for observed Yes ",
    "and %.3f for observed No."
  ),
  unname(mean_probability_by_outcome["Yes"]),
  unname(mean_probability_by_outcome["No"])
))

# =============================================================================
# 3. Probability-scale FastSHAP for all 621 training patients
# =============================================================================

set.seed(20260904)
shap_values <- fastshap::explain(
  object = locked_xgboost,
  X = X,
  newdata = X,
  pred_wrapper = pred_success,
  nsim = shap_nsim,
  adjust = TRUE,
  shap_only = TRUE
)

stopifnot(
  nrow(shap_values) == nrow(X),
  identical(colnames(shap_values), colnames(X))
)

shap_baseline <- attr(shap_values, "baseline")
if (is.null(shap_baseline) || length(shap_baseline) != 1L ||
    !is.finite(shap_baseline)) {
  stop("FastSHAP did not return a valid probability-scale baseline.")
}

# adjust = TRUE should enforce local accuracy on the probability scale.
reconstructed_probability <- shap_baseline + rowSums(shap_values)
local_accuracy_error <- p_success - reconstructed_probability
max_local_accuracy_error <- max(abs(local_accuracy_error))

message(sprintf(
  "Maximum probability-scale local-accuracy error: %.3g",
  max_local_accuracy_error
))

if (max_local_accuracy_error > 1e-6) {
  warning(
    "Maximum local-accuracy error exceeds 1e-6; inspect the audit CSV before use."
  )
}

# =============================================================================
# 4. Clinically readable labels and audit tables
# =============================================================================

feature_labels <- c(
  Stone_volume = "Stone volume (mm\u00b3)",
  Mean_CT_attenuation = "Mean CT attenuation (HU)",
  Ureteral_wall_thickness = "Ureteral wall thickness (mm)",
  Hydronephrosis_measure = "Hydronephrosis measurement (mm)",
  Subcutaneous_fat_thickness = "Subcutaneous fat thickness (mm)",
  Stone_location = "Stone location"
)

# Short labels prevent unreadably small text when the combined figure is
# displayed at a journal-standard full-page width of approximately 180 mm.
feature_labels_short <- c(
  Stone_volume = "SV (mm\u00b3)",
  Mean_CT_attenuation = "MeCT (HU)",
  Ureteral_wall_thickness = "UWT (mm)",
  Hydronephrosis_measure = "Hydronephrosis (mm)",
  Subcutaneous_fat_thickness = "SFT (mm)",
  Stone_location = "Stone location"
)

X_display <- X
names(X_display) <- unname(feature_labels[names(X_display)])

shap_display <- shap_values
colnames(shap_display) <- unname(feature_labels[colnames(shap_display)])

shp <- shapviz::shapviz(
  shap_display,
  X = X_display,
  baseline = shap_baseline
)

# A second shapviz object with compact labels is used only for the individual
# waterfall panels. The underlying SHAP values are identical.
X_individual <- X
names(X_individual) <- unname(feature_labels_short[names(X_individual)])

shap_individual <- shap_values
colnames(shap_individual) <- unname(
  feature_labels_short[colnames(shap_individual)]
)

shp_individual <- shapviz::shapviz(
  shap_individual,
  X = X_individual,
  baseline = shap_baseline
)

importance_table <- tibble(
  feature_code = colnames(shap_values),
  feature = unname(feature_labels[colnames(shap_values)]),
  mean_absolute_SHAP = colMeans(abs(shap_values)),
  median_absolute_SHAP = apply(abs(shap_values), 2L, median)
) %>%
  arrange(desc(mean_absolute_SHAP)) %>%
  mutate(rank = row_number(), .before = 1L)

local_accuracy_table <- tibble(
  Case_ID = training_model$Case_ID,
  observed_composite_success = training_model$success,
  predicted_success_probability = p_success,
  SHAP_baseline = shap_baseline,
  reconstructed_probability = reconstructed_probability,
  reconstruction_error = local_accuracy_error
)

# Select cases nearest the pre-specified probability quantiles. Selection does
# not use observed outcome and is therefore deterministic rather than subjective.
representative_ids <- vapply(
  representative_quantiles,
  function(q) which.min(abs(p_success - unname(quantile(p_success, q, names = FALSE)))),
  integer(1)
)

representative_table <- tibble(
  explanation = names(representative_ids),
  probability_quantile = unname(representative_quantiles),
  row_id = unname(representative_ids),
  Case_ID = training_model$Case_ID[representative_ids],
  observed_composite_success = training_model$success[representative_ids],
  predicted_success_probability = p_success[representative_ids]
) %>%
  bind_cols(X[representative_ids, , drop = FALSE])

write_csv(
  bind_cols(tibble(Case_ID = training_model$Case_ID), as_tibble(shap_values)),
  file.path(output_dir, "SHAP_training_values_probability_scale.csv")
)
write_csv(
  importance_table,
  file.path(output_dir, "SHAP_global_importance_training.csv")
)
write_csv(
  local_accuracy_table,
  file.path(output_dir, "SHAP_local_accuracy_audit.csv")
)
write_csv(
  representative_table,
  file.path(output_dir, "SHAP_representative_patients.csv")
)

saveRDS(
  list(
    shap_values = shap_values,
    shapviz_object = shp,
    X = X,
    predicted_success_probability = p_success,
    baseline = shap_baseline,
    importance = importance_table,
    representative_patients = representative_table,
    seed = 20260904L,
    nsim = shap_nsim,
    target_probability = target_col,
    cohort = "training"
  ),
  file.path(output_dir, "SHAP_training_complete_object.rds")
)

# =============================================================================
# 5. Publication theme
# =============================================================================

theme_publication <- function(base_size = 8.2) {
  theme_bw(base_size = base_size, base_family = base_family) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.25, colour = "#E7E7E7"),
      panel.border = element_rect(linewidth = 0.55, colour = "#333333"),
      axis.title = element_text(colour = "#111111"),
      axis.text = element_text(colour = "#333333"),
      plot.title = element_text(face = "bold", size = rel(1.06), hjust = 0),
      plot.subtitle = element_text(size = rel(0.92), colour = "#4D4D4D", hjust = 0),
      plot.caption = element_text(size = rel(0.76), colour = "#5C5C5C", hjust = 0),
      plot.margin = margin(7, 9, 7, 7),
      legend.title = element_text(face = "bold"),
      legend.key.height = grid::unit(12, "mm")
    )
}

shap_x_label <- "SHAP value for P(composite ESWL success)"
shap_x_label_short <- "SHAP value for P(success)"

# =============================================================================
# 6. Main figure: global effect plus three pre-specified individual examples
# =============================================================================

# Construct the beeswarm manually so the categorical Stone_location predictor
# is shown in grey instead of being misleadingly placed on a continuous
# low-to-high color scale. Continuous values are normalized within each feature
# for color display only; the model inputs and SHAP values are not standardized.
normalize_continuous_for_color <- function(z) {
  z <- as.numeric(z)
  robust_limits <- as.numeric(
    quantile(z, probs = c(0.05, 0.95), na.rm = TRUE, names = FALSE)
  )

  if (!all(is.finite(robust_limits)) || diff(robust_limits) <= 0) {
    return(rep(0.5, length(z)))
  }

  scales::squish(
    scales::rescale(z, from = robust_limits, to = c(0, 1)),
    range = c(0, 1)
  )
}

global_plot_data <- bind_rows(lapply(predictor_names, function(variable) {
  normalized_value <- if (is.numeric(X[[variable]])) {
    normalize_continuous_for_color(X[[variable]])
  } else {
    # NA is deliberately mapped to grey by the color scale below.
    rep(NA_real_, nrow(X))
  }

  tibble(
    feature_code = variable,
    feature = unname(feature_labels_short[variable]),
    shap_value = shap_values[, variable],
    normalized_feature_value = normalized_value
  )
}))

feature_order <- importance_table$feature_code
global_plot_data$feature <- factor(
  global_plot_data$feature,
  levels = rev(unname(feature_labels_short[feature_order]))
)

shap_range <- range(global_plot_data$shap_value, finite = TRUE)
shap_span <- diff(shap_range)
if (!is.finite(shap_span) || shap_span <= 0) shap_span <- 1

importance_label_position <- shap_range[1] - 0.075 * shap_span
global_y_limits <- c(
  shap_range[1] - 0.16 * shap_span,
  shap_range[2] + 0.025 * shap_span
)

importance_plot_labels <- importance_table %>%
  transmute(
    feature = factor(
      unname(feature_labels_short[feature_code]),
      levels = levels(global_plot_data$feature)
    ),
    shap_value = importance_label_position,
    label = sprintf("%.4f", mean_absolute_SHAP)
  )

pA <- ggplot(
  global_plot_data,
  aes(
    x = feature,
    y = shap_value,
    colour = normalized_feature_value
  )
) +
  geom_hline(
    yintercept = 0,
    linewidth = 0.42,
    colour = "#666666"
  ) +
  ggbeeswarm::geom_quasirandom(
    groupOnX = TRUE,
    width = 0.34,
    size = 0.72,
    alpha = 0.78,
    show.legend = TRUE
  ) +
  geom_text(
    data = importance_plot_labels,
    aes(x = feature, y = shap_value, label = label),
    inherit.aes = FALSE,
    family = base_family,
    size = 2.35,
    colour = "#222222",
    hjust = 0.5
  ) +
  coord_flip(ylim = global_y_limits, clip = "off") +
  scale_colour_gradient(
    low = "#2C7BB6",
    high = "#D7191C",
    limits = c(0, 1),
    breaks = c(0, 1),
    labels = c("Low", "High"),
    na.value = "#777777",
    name = "Normalized\nfeature value",
    guide = guide_colorbar(
      title.position = "top",
      barheight = grid::unit(18, "mm"),
      barwidth = grid::unit(3.0, "mm")
    )
  ) +
  labs(
    title = "Global SHAP",
    subtitle = "Numbers: mean |SHAP|; grey: stone location",
    x = NULL,
    y = shap_x_label_short
  ) +
  theme_publication(8.2) +
  theme(
    legend.position = "right",
    legend.title = element_text(size = 6.8, face = "bold"),
    legend.text = element_text(size = 6.5),
    axis.text.y = element_text(size = 7.2)
  )

id_low <- unname(representative_ids["lower"])
id_mid <- unname(representative_ids["median"])
id_high <- unname(representative_ids["higher"])

pB <- shapviz::sv_waterfall(
  shp_individual,
  row_id = id_mid,
  max_display = length(predictor_names)
) +
  labs(
    title = NULL,
    x = "Prediction",
    y = NULL
  ) +
  theme_publication(8.2)

pC <- shapviz::sv_waterfall(
  shp_individual,
  row_id = id_low,
  max_display = length(predictor_names)
) +
  labs(
    title = NULL,
    x = "Prediction",
    y = NULL
  ) +
  theme_publication(8.2)

pD <- shapviz::sv_waterfall(
  shp_individual,
  row_id = id_high,
  max_display = length(predictor_names)
) +
  labs(
    title = NULL,
    x = "Prediction",
    y = NULL
  ) +
  theme_publication(8.2)

figure_main <- patchwork::wrap_plots(
  pA, pB, pC, pD,
  ncol = 2,
  widths = c(1.05, 1),
  heights = c(1, 1)
) +
  patchwork::plot_annotation(
    tag_levels = "A",
    theme = theme(
      plot.tag = element_text(
        family = base_family,
        face = "bold",
        size = 11,
        colour = "#111111"
      ),
      plot.caption = element_text(
        family = base_family,
        size = 6.4,
        colour = "#555555",
        hjust = 0,
        lineheight = 1.05,
        margin = margin(t = 4)
      )
    )
  )

# =============================================================================
# 7. Dependence/effect figure for all six predictors
# =============================================================================

continuous_features <- predictor_names[predictor_names != "Stone_location"]

make_continuous_dependence_plot <- function(variable) {
  display_limits <- as.numeric(
    quantile(X[[variable]], probs = c(0.01, 0.99), na.rm = TRUE, names = FALSE)
  )

  plot_data <- tibble(
    predictor_value = X[[variable]],
    shap_value = shap_values[, variable]
  )

  ggplot(plot_data, aes(x = predictor_value, y = shap_value)) +
    geom_hline(yintercept = 0, linewidth = 0.45, linetype = "dashed",
               colour = "#6A6A6A") +
    geom_point(shape = 16, size = 1.35, alpha = 0.55, colour = "#2C6A9B") +
    geom_smooth(
      method = "loess",
      formula = y ~ x,
      span = 0.80,
      se = TRUE,
      linewidth = 0.80,
      colour = "#B2182B",
      fill = "#F4A582",
      alpha = 0.22
    ) +
    coord_cartesian(xlim = display_limits) +
    labs(
      title = unname(feature_labels[variable]),
      x = unname(feature_labels_short[variable]),
      y = shap_x_label_short
    ) +
    theme_publication(8.2)
}

dependence_plots <- lapply(continuous_features, make_continuous_dependence_plot)

location_plot_data <- tibble(
  Stone_location = X$Stone_location,
  shap_value = shap_values[, "Stone_location"]
)

p_location <- ggplot(
  location_plot_data,
  aes(x = Stone_location, y = shap_value, fill = Stone_location)
) +
  geom_hline(yintercept = 0, linewidth = 0.45, linetype = "dashed",
             colour = "#6A6A6A") +
  geom_boxplot(
    width = 0.58,
    outlier.shape = NA,
    linewidth = 0.55,
    alpha = 0.70
  ) +
  geom_jitter(
    aes(colour = Stone_location),
    width = 0.13,
    height = 0,
    size = 1.05,
    alpha = 0.38,
    show.legend = FALSE
  ) +
  scale_fill_manual(values = c(
    "Distal ureter" = "#4E79A7",
    "Mid ureter" = "#F28E2B",
    "Proximal ureter" = "#59A14F"
  )) +
  scale_colour_manual(values = c(
    "Distal ureter" = "#2F5D8A",
    "Mid ureter" = "#C66C12",
    "Proximal ureter" = "#387A32"
  )) +
  scale_x_discrete(labels = c(
    "Distal ureter" = "Distal",
    "Mid ureter" = "Mid",
    "Proximal ureter" = "Proximal"
  )) +
  labs(
    title = unname(feature_labels["Stone_location"]),
    x = "Ureteral stone location",
    y = shap_x_label_short
  ) +
  guides(fill = "none") +
  theme_publication(8.2)

dependence_plots <- c(dependence_plots, list(p_location))

figure_dependence <- patchwork::wrap_plots(
  dependence_plots,
  ncol = 2
) +
  patchwork::plot_annotation(
    tag_levels = "A",
    caption = paste0(
      "Continuous-variable axes show the 1st-99th percentile for readability; ",
      "SHAP values and loess curves were calculated using all training observations."
    ),
    theme = theme(
      plot.tag = element_text(
        family = base_family,
        face = "bold",
        size = 14,
        colour = "#111111"
      ),
      plot.caption = element_text(
        family = base_family,
        size = 8,
        colour = "#555555",
        hjust = 0,
        margin = margin(t = 6)
      )
    )
  )

# =============================================================================
# 8. Export vector and high-resolution raster files
# =============================================================================

export_figure <- function(plot_object, stem, width, height) {
  ggsave(
    filename = file.path(output_dir, paste0(stem, ".pdf")),
    plot = plot_object,
    device = grDevices::cairo_pdf,
    width = width,
    height = height,
    units = "in",
    bg = "white"
  )

  ggsave(
    filename = file.path(output_dir, paste0(stem, ".png")),
    plot = plot_object,
    width = width,
    height = height,
    units = "in",
    dpi = 600,
    bg = "white"
  )

  ggsave(
    filename = file.path(output_dir, paste0(stem, ".tiff")),
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
  figure_main,
  "Figure_SHAP_global_and_individual",
  width = 7.2,
  height = 5.5
)

export_figure(
  figure_dependence,
  "Figure_SHAP_dependence_all_6_predictors",
  width = 7.2,
  height = 8.2
)

writeLines(
  capture.output(sessionInfo()),
  con = file.path(output_dir, "SHAP_sessionInfo.txt")
)

message(
  "Completed. Primary explanation cohort: training (n = ", nrow(X), ").\n",
  "Target probability: ", target_col, " (composite ESWL success).\n",
  "Outputs: ", normalizePath(output_dir, winslash = "/", mustWork = TRUE)
)
