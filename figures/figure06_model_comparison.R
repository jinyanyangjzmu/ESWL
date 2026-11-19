# ==============================================================================
# Figure 6: comparison of nine algorithms in the development cohort
# Retained content: repeated-CV AUC, OOF Brier score, calibration intercept,
# and calibration slope. The sealed test cohort is never read.
# ==============================================================================




# 1. Paths --------------------------------------------------------------------

results_dir <- paste0(
  "output/",
  "model_benchmark_9_CV_FORMAL"
)

figure_dir <- file.path(results_dir, "model_comparison_visualization")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

input_files <- c(
  auc     = file.path(results_dir, "01_development_CV_AUC_values.csv"),
  summary = file.path(results_dir, "02_development_CV_model_summary.csv"),
  config  = file.path(results_dir, "06_run_configuration.csv")
)

if (any(!file.exists(input_files))) {
  stop(
    "Missing input file(s):\n",
    paste(input_files[!file.exists(input_files)], collapse = "\n")
  )
}


# 2. Packages and data --------------------------------------------------------

required_packages <- c(
  "ggplot2", "dplyr", "readr", "patchwork", "scales"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop("Please install: ", paste(missing_packages, collapse = ", "))
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(patchwork)
  library(scales)
})

cv_auc <- read_csv(input_files[["auc"]], show_col_types = FALSE)
model_summary <- read_csv(input_files[["summary"]], show_col_types = FALSE)
run_config <- read_csv(
  input_files[["config"]],
  show_col_types = FALSE,
  col_types = cols(.default = col_character())
)


# 3. Checks, labels, order, and colors ----------------------------------------

check_columns <- function(data, required, file_name) {
  missing <- setdiff(required, names(data))
  if (length(missing) > 0L) {
    stop(file_name, " is missing: ", paste(missing, collapse = ", "))
  }
}

check_columns(
  cv_auc,
  c("Model", "Repeat", "Fold", "Resample", "AUC"),
  basename(input_files[["auc"]])
)

check_columns(
  model_summary,
  c(
    "Model", "n_resamples", "CV_AUC_mean", "CV_AUC_SD",
    "OOF_Brier", "Calibration_intercept", "Calibration_slope",
    "Tuning_notes_n", "Complete"
  ),
  basename(input_files[["summary"]])
)

check_columns(
  run_config,
  c("Item", "Value"),
  basename(input_files[["config"]])
)

config_value <- function(item) {
  value <- run_config$Value[match(item, run_config$Item)]
  if (length(value) != 1L || is.na(value)) stop("Missing config item: ", item)
  value
}

if (config_value("Run_mode") != "FORMAL_CV" ||
    toupper(config_value("Heldout_cohort_extracted_or_evaluated")) != "FALSE") {
  stop("Figure 6 requires the FORMAL_CV run with the held-out cohort sealed.")
}

cv_folds <- as.integer(config_value("CV_folds"))
cv_repeats <- as.integer(config_value("CV_repeats"))
expected_resamples <- cv_folds * cv_repeats

model_names <- c(
  "Logistic", "DT", "RF", "XGBoost", "RSVM",
  "MLP", "LightGBM", "ENet", "kNN"
)

if (!setequal(unique(cv_auc$Model), model_names) ||
    !setequal(unique(model_summary$Model), model_names)) {
  stop("The input tables must contain the expected nine models.")
}

if (any(cv_auc %>% count(Model) %>% pull(n) != expected_resamples) ||
    anyDuplicated(cv_auc[c("Model", "Resample")])) {
  stop("Fold-level AUC values are incomplete or duplicated.")
}

if (any(!model_summary$Complete) || any(model_summary$Tuning_notes_n > 0L)) {
  stop("At least one model is incomplete or has an unresolved tuning note.")
}

model_labels <- c(
  Logistic = "Logistic regression",
  DT       = "Decision tree",
  RF       = "Random forest",
  XGBoost  = "XGBoost",
  RSVM     = "RBF-SVM",
  MLP      = "Multilayer perceptron",
  LightGBM = "LightGBM",
  ENet     = "Elastic net",
  kNN      = "k-nearest neighbors"
)

# Same nine-model palette as the preceding visualization.
model_colors <- c(
  Logistic = "#4E79A7",
  DT       = "#A0CBE8",
  RF       = "#59A14F",
  XGBoost  = "#F28E2B",
  RSVM     = "#B07AA1",
  MLP      = "#E15759",
  LightGBM = "#76B7B2",
  ENet     = "#EDC948",
  kNN      = "#9C755F"
)

# One fixed order is used in all four panels.
model_order <- model_summary %>%
  arrange(desc(CV_AUC_mean), Model) %>%
  pull(Model)

cv_auc <- cv_auc %>%
  mutate(Model = factor(Model, levels = rev(model_order)))

model_summary <- model_summary %>%
  mutate(Model = factor(Model, levels = rev(model_order)))


# 4. Publication theme and helpers --------------------------------------------

theme_publication <- theme_classic(base_size = 8.5, base_family = "Arial") +
  theme(
    plot.title = element_text(
      face = "bold", size = 9.2, color = "#202020",
      margin = margin(b = 2.5)
    ),
    plot.subtitle = element_text(
      size = 7.4, color = "#555555", margin = margin(b = 5)
    ),
    axis.title = element_text(face = "bold", size = 8.2),
    axis.text = element_text(size = 7.6, color = "#303030"),
    axis.ticks = element_line(linewidth = 0.35, color = "#555555"),
    axis.line = element_line(linewidth = 0.40, color = "#555555"),
    panel.grid.major.x = element_line(linewidth = 0.25, color = "#E5E5E5"),
    panel.grid.minor = element_blank(),
    legend.position = "none",
    plot.margin = margin(5, 7, 5, 5)
  )

metric_plot <- function(data, value, ideal, title, x_label) {
  value_data <- data %>%
    transmute(Model, Value = .data[[value]])

  ggplot(value_data, aes(y = Model, color = Model)) +
    geom_vline(
      xintercept = ideal, linetype = "dashed",
      linewidth = 0.45, color = "#707070"
    ) +
    geom_segment(
      aes(x = ideal, xend = Value, yend = Model),
      linewidth = 0.55, alpha = 0.65, lineend = "round"
    ) +
    geom_point(aes(x = Value), size = 2.35, stroke = 0) +
    scale_color_manual(values = model_colors, drop = FALSE) +
    scale_y_discrete(labels = model_labels, drop = FALSE) +
    scale_x_continuous(
      labels = label_number(accuracy = 0.01),
      breaks = pretty_breaks(n = 5),
      expand = expansion(mult = c(0.08, 0.12))
    ) +
    labs(title = title, x = x_label, y = NULL) +
    theme_publication
}


# 5A. Fold-to-fold CV AUC -----------------------------------------------------

auc_limits <- c(
  max(0, floor((min(cv_auc$AUC, na.rm = TRUE) - 0.01) / 0.05) * 0.05),
  min(1, ceiling((max(cv_auc$AUC, na.rm = TRUE) + 0.01) / 0.05) * 0.05)
)

p_auc <- ggplot(cv_auc, aes(x = AUC, y = Model, color = Model)) +
  geom_boxplot(
    aes(fill = Model), width = 0.58, linewidth = 0.45,
    alpha = 0.18, outlier.shape = NA
  ) +
  geom_point(
    position = position_jitter(width = 0, height = 0.105, seed = 20260912),
    size = 0.72, alpha = 0.34, stroke = 0
  ) +
  stat_summary(
    fun = mean, geom = "point", shape = 23,
    size = 2.25, stroke = 0.60, fill = "white"
  ) +
  scale_color_manual(values = model_colors, drop = FALSE) +
  scale_fill_manual(values = model_colors, drop = FALSE) +
  scale_y_discrete(labels = model_labels, drop = FALSE) +
  scale_x_continuous(
    limits = auc_limits,
    breaks = breaks_width(0.05),
    labels = label_number(accuracy = 0.01)
  ) +
  labs(
    title = "Fold-to-fold discrimination",
    x = "ROC AUC", y = NULL
  ) +
  theme_publication


# 5B-D. OOF probability performance ------------------------------------------

p_brier <- metric_plot(
  model_summary, "OOF_Brier", ideal = 0,
  title = "OOF Brier score",
  x_label = "Brier score"
)

p_intercept <- metric_plot(
  model_summary, "Calibration_intercept", ideal = 0,
  title = "Calibration intercept",
  x_label = "Intercept"
)

p_slope <- metric_plot(
  model_summary, "Calibration_slope", ideal = 1,
  title = "Calibration slope",
  x_label = "Slope"
)

# Model labels are shown in the left column and shared by the aligned right
# column, preventing repetition while retaining exact model identification.
hide_right_y <- theme(
  axis.text.y = element_blank(),
  axis.ticks.y = element_blank(),
  axis.line.y = element_blank()
)

p_brier <- p_brier + hide_right_y
p_slope <- p_slope + hide_right_y


# 6. Assemble and export -------------------------------------------------------

# The retained original A/C/D/E content is renumbered consecutively A-D.
figure6 <-
  (p_auc | p_brier) /
  (p_intercept | p_slope) +
  plot_layout(widths = c(1.08, 0.92), heights = c(1, 1)) +
  plot_annotation(
    tag_levels = "A",
    theme = theme(
      plot.tag = element_text(
        family = "Arial", face = "bold", size = 11, color = "#111111"
      ),
      plot.tag.position = c(0.01, 0.995),
      plot.margin = margin(4, 4, 4, 4)
    )
  )

print(figure6)

ggsave(
  file.path(figure_dir, "Figure6_9_model_comparison.tiff"),
  plot = figure6,
  width = 180, height = 165, units = "mm",
  dpi = 600, compression = "lzw", bg = "white"
)

pdf_file <- file.path(figure_dir, "Figure6_9_model_comparison.pdf")
tryCatch(
  ggsave(
    pdf_file, plot = figure6,
    width = 180, height = 165, units = "mm",
    device = cairo_pdf, bg = "white"
  ),
  error = function(e) {
    ggsave(
      pdf_file, plot = figure6,
      width = 180, height = 165, units = "mm",
      device = "pdf", bg = "white"
    )
  }
)
