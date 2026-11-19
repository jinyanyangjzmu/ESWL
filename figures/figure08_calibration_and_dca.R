# Figure 8: locked XGBoost calibration and decision curves (three cohorts)
# One combined figure only: vector PDF and 600-dpi TIFF.
# The original model probabilities are never recalibrated or refitted.

# 1. Paths and settings -------------------------------------------------------
data_file <- paste0(
  "data/",
  "复合碎石成功的三个队列.RData"
)
model_file <- paste0(
  "output/",
  "model_benchmark_9_CV_FORMAL/13_final_models/",
  "04_XGBoost_best_CV_refit_621.rds"
)
output_dir <- paste0(
  "output/",
  "figure8_locked_xgboost_calibration_dca"
)

cohort_names <- c(
  "Development cohort", "Internal test cohort", "External validation cohort"
)
cohort_titles <- c(
  "Development cohort" = "Training set (apparent)",
  "Internal test cohort" = "Internal test set",
  "External validation cohort" = "External validation set"
)
bin_counts <- c(5L, 5L, 3L)
thresholds <- seq(0.10, 0.90, by = 0.005)

required <- c("tidymodels", "dplyr", "tidyr", "purrr", "ggplot2",
              "cowplot", "scales", "xgboost")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Install missing packages: ", paste(missing, collapse = ", "))
suppressPackageStartupMessages({
  library(tidymodels)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(ggplot2)
  library(cowplot)
})
if (!all(file.exists(c(data_file, model_file)))) stop("Data or model file is missing.")

# Keep the same audited inputs as the original analysis.
if (unname(tools::md5sum(data_file)) != "3b085c679b1d23aa6bc17c9bcefd07bb" ||
    unname(tools::md5sum(model_file)) != "25e87ace906a55706bda38e6ec29e8b2") {
  stop("Input file differs from the audited data/model; check the paths.")
}

# 2. Original six-predictor input and locked probabilities --------------------
input <- new.env(parent = emptyenv())
loaded <- load(data_file, envir = input)
if (!all(c("training_data", "test_data", "external_data") %in% loaded)) {
  stop("RData must contain training_data, test_data, and external_data.")
}

as_numeric <- function(x, label) {
  result <- suppressWarnings(as.numeric(trimws(as.character(x))))
  if (any(!is.finite(result))) stop("Invalid values in ", label)
  result
}

prepare <- function(data, cohort) {
  needed <- c("success", "Stone volume", "Mean CT attenuation",
              "Ureteral wall thickness", "hydronephrosis",
              "Subcutaneous fat thickness", "Stone location")
  if (!all(needed %in% names(data))) stop(cohort, ": missing model input.")

  y <- tolower(trimws(as.character(data$success)))
  y[y %in% c("yes", "1", "success", "successful")] <- "1"
  y[y %in% c("no", "0", "failure", "failed", "unsuccessful")] <- "0"
  if (anyNA(y) || !all(y %in% c("1", "0"))) stop(cohort, ": invalid success.")

  location <- tolower(trimws(gsub("[[:space:]]+", " ",
                                  as.character(data[["Stone location"]]))))
  location <- unname(c(
    "distal ureter" = "Distal ureter", "distal" = "Distal ureter",
    "mid ureter" = "Mid ureter", "middle ureter" = "Mid ureter",
    "mid" = "Mid ureter", "middle" = "Mid ureter",
    "proximal ureter" = "Proximal ureter", "proximal" = "Proximal ureter"
  )[location])
  if (anyNA(location)) stop(cohort, ": invalid Stone location.")

  tibble::tibble(
    Case_ID = seq_len(nrow(data)),
    success = factor(y, levels = c("1", "0")),
    Stone_volume = as_numeric(data[["Stone volume"]], "Stone volume"),
    Mean_CT_attenuation = as_numeric(data[["Mean CT attenuation"]], "Mean CT attenuation"),
    Ureteral_wall_thickness = as_numeric(data[["Ureteral wall thickness"]], "Ureteral wall thickness"),
    Hydronephrosis_measure = as_numeric(data[["hydronephrosis"]], "hydronephrosis"),
    Subcutaneous_fat_thickness = as_numeric(data[["Subcutaneous fat thickness"]], "Subcutaneous fat thickness"),
    Stone_location = factor(
      location, levels = c("Distal ureter", "Mid ureter", "Proximal ureter")
    )
  )
}

raw_names <- c("training_data", "test_data", "external_data")
cohorts <- setNames(
  lapply(seq_along(raw_names), function(i) {
    prepare(get(raw_names[i], envir = input, inherits = FALSE), cohort_names[i])
  }),
  cohort_names
)
if (!identical(unname(vapply(cohorts, nrow, integer(1))), c(621L, 208L, 147L)) ||
    !identical(unname(vapply(cohorts, function(x) sum(x$success == "1"), integer(1))),
               c(520L, 174L, 134L))) {
  stop("Cohort size/outcome counts differ from the original Figure 8 analysis.")
}

locked_model <- readRDS(model_file)
predictions <- purrr::imap_dfr(cohorts, function(data, name) {
  probability <- predict(locked_model, new_data = data, type = "prob")$.pred_1
  if (length(probability) != nrow(data) || any(!is.finite(probability)) ||
      any(probability < 0 | probability > 1)) {
    stop(name, ": invalid locked-model probabilities.")
  }
  tibble::tibble(Cohort = name, truth = as.integer(data$success == "1"),
                 .pred_1 = as.numeric(probability))
}) |>
  mutate(Cohort = factor(Cohort, levels = cohort_names))
by_cohort <- split(predictions, predictions$Cohort)
safe_logit <- function(p) qlogis(pmin(pmax(p, 1e-6), 1 - 1e-6))

# 3. Plot-specific calibration and decision-curve data ------------------------
calibration <- purrr::imap(by_cohort, function(data, name) {
  data$lp <- safe_logit(data$.pred_1)
  intercept_fit <- suppressWarnings(glm(
    truth ~ 1 + offset(lp), data = data, family = binomial()
  ))
  curve_fit <- suppressWarnings(glm(truth ~ lp, data = data, family = binomial()))
  intercept <- unname(coef(intercept_fit)[1])
  slope <- unname(coef(curve_fit)["lp"])
  if (any(!is.finite(c(intercept, slope)))) stop(name, ": invalid calibration fit.")

  grid <- tibble::tibble(.pred_1 = seq(min(data$.pred_1), max(data$.pred_1),
                                       length.out = 250L)) |>
    mutate(lp = safe_logit(.pred_1))
  fitted <- predict(curve_fit, newdata = grid, type = "link", se.fit = TRUE)
  grid <- grid |>
    mutate(Observed = plogis(fitted$fit),
           Lower = plogis(fitted$fit - 1.96 * fitted$se.fit),
           Upper = plogis(fitted$fit + 1.96 * fitted$se.fit))

  n_bins <- bin_counts[match(name, cohort_names)]
  breaks <- unique(as.numeric(quantile(
    data$.pred_1, probs = seq(0, 1, length.out = n_bins + 1L),
    names = FALSE, type = 8
  )))
  if (length(breaks) < 3L) stop(name, ": insufficient unique probabilities.")
  points <- data |>
    mutate(Group = cut(.pred_1, breaks = breaks, include.lowest = TRUE,
                       labels = FALSE)) |>
    group_by(Group) |>
    summarise(N = n(), Events = sum(truth), Mean_predicted = mean(.pred_1),
              Observed = mean(truth), .groups = "drop")
  exact_ci <- t(vapply(seq_len(nrow(points)), function(i) {
    unname(binom.test(points$Events[i], points$N[i])$conf.int)
  }, numeric(2)))
  points$Lower <- exact_ci[, 1]
  points$Upper <- exact_ci[, 2]

  list(
    curve = grid,
    points = points,
    subtitle = sprintf("CITL %.2f; slope %.2f; Brier %.3f",
                       intercept, slope, mean((data$.pred_1 - data$truth)^2))
  )
})

dca <- purrr::imap_dfr(by_cohort, function(data, name) {
  prevalence <- mean(data$truth)
  model_benefit <- vapply(thresholds, function(t) {
    positive <- data$.pred_1 >= t
    mean(positive & data$truth == 1L) -
      mean(positive & data$truth == 0L) * t / (1 - t)
  }, numeric(1))
  tibble::tibble(
    Cohort = name, Threshold = thresholds, XGBoost = model_benefit,
    `Treat all` = prevalence - (1 - prevalence) * thresholds / (1 - thresholds),
    `Treat none` = 0
  ) |>
    pivot_longer(c("XGBoost", "Treat all", "Treat none"),
                 names_to = "Strategy", values_to = "Net_benefit")
}) |>
  mutate(Strategy = factor(Strategy, levels = c("XGBoost", "Treat all", "Treat none")))

# 4. Six publication-sized panels and one visible DCA legend -----------------
palette <- c(XGBoost = "#5547A7", `Treat all` = "#D06467",
             `Treat none` = "#667085")
line_types <- c(XGBoost = "solid", `Treat all` = "longdash",
                `Treat none` = "dotted")
paper_theme <- theme_bw(base_size = 9, base_family = "Arial") +
  theme(
    plot.title = element_text(face = "bold", size = 9, hjust = 0.5,
                              margin = ggplot2::margin(b = 3)),
    plot.subtitle = element_text(size = 7.4, hjust = 0.5,
                                 margin = ggplot2::margin(b = 4)),
    axis.title = element_text(size = 8),
    axis.text = element_text(size = 7.4, color = "#30343B"),
    panel.grid.major = element_line(color = "#E3E5EA", linewidth = 0.30),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "#363B42", linewidth = 0.45),
    plot.margin = ggplot2::margin(5, 5, 4, 5)
  )

calibration_panel <- function(name) {
  info <- calibration[[name]]
  subtitle <- info$subtitle
  if (name == cohort_names[3]) subtitle <- paste0(
    subtitle, "\n13 failures; wide uncertainty"
  )
  ggplot() +
    geom_abline(intercept = 0, slope = 1, color = "#7D838C",
                linetype = "dashed", linewidth = 0.6) +
    geom_ribbon(data = info$curve,
                aes(x = .pred_1, ymin = Lower, ymax = Upper),
                fill = "#C33C54", alpha = 0.16) +
    geom_line(data = info$curve, aes(x = .pred_1, y = Observed),
              color = "#B7354D", linewidth = 0.85) +
    geom_errorbar(data = info$points,
                  aes(x = Mean_predicted, ymin = Lower, ymax = Upper),
                  width = 0.018, color = "#2E6EB5", linewidth = 0.55) +
    geom_point(data = info$points, aes(x = Mean_predicted, y = Observed),
               shape = 21, size = 2.4, stroke = 0.4,
               fill = "#2E6EB5", color = "white") +
    geom_rug(data = by_cohort[[name]], aes(x = .pred_1),
             sides = "b", length = grid::unit(0.02, "npc"),
             color = "#5C6673", alpha = 0.22, linewidth = 0.2) +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25),
                       labels = scales::label_number(accuracy = 0.01),
                       expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25),
                       labels = scales::label_number(accuracy = 0.01),
                       expand = c(0, 0)) +
    coord_equal() +
    labs(title = cohort_titles[[name]], subtitle = subtitle,
         x = "Predicted probability",
         y = if (name == cohort_names[1]) "Observed success proportion" else NULL) +
    paper_theme
}

dca_panel <- function(name, legend = FALSE) {
  ggplot(filter(dca, Cohort == name),
         aes(x = Threshold, y = Net_benefit,
             color = Strategy, linetype = Strategy)) +
    geom_line(linewidth = 0.95) +
    scale_color_manual(values = palette, breaks = names(palette), drop = FALSE) +
    scale_linetype_manual(values = line_types, breaks = names(line_types),
                          drop = FALSE) +
    guides(color = guide_legend(nrow = 1,
             override.aes = list(linetype = unname(line_types), linewidth = 1.4)),
           linetype = "none") +
    scale_x_continuous(limits = c(0.10, 0.90),
                       breaks = seq(0.10, 0.90, 0.10),
                       labels = scales::label_number(accuracy = 0.1),
                       expand = c(0, 0)) +
    scale_y_continuous(breaks = seq(0, 0.9, 0.1),
                       labels = scales::label_number(accuracy = 0.1)) +
    coord_cartesian(ylim = c(-0.05, 0.95)) +
    labs(title = cohort_titles[[name]], x = "Threshold probability",
         y = if (name == cohort_names[1]) "Net benefit" else NULL,
         color = NULL, linetype = NULL) +
    paper_theme +
    theme(legend.position = if (legend) "bottom" else "none",
          legend.direction = "horizontal",
          legend.justification = "center",
          legend.key.width = grid::unit(1.2, "cm"),
          legend.text = element_text(size = 8.5))
}

# Draw the DCA key as a dedicated strip so it remains visible after export.
key_data <- tibble::tibble(
  Strategy = factor(names(palette), levels = names(palette)),
  x0 = c(0.11, 0.40, 0.69),
  x1 = c(0.18, 0.47, 0.76),
  label_x = c(0.20, 0.49, 0.78)
)
legend <- ggplot(key_data) +
  geom_segment(aes(x = x0, xend = x1, y = 0.5, yend = 0.5,
                   color = Strategy, linetype = Strategy), linewidth = 1.4) +
  geom_text(aes(x = label_x, y = 0.5, label = Strategy),
            hjust = 0, size = 3.1, color = "#20242A", family = "Arial") +
  scale_color_manual(values = palette, guide = "none") +
  scale_linetype_manual(values = line_types, guide = "none") +
  xlim(0, 1) + ylim(0, 1) + theme_void()
top_row <- cowplot::plot_grid(plotlist = lapply(cohort_names, calibration_panel),
                             nrow = 1, align = "hv", axis = "tblr")
bottom_row <- cowplot::plot_grid(plotlist = lapply(cohort_names, dca_panel),
                                nrow = 1, align = "hv", axis = "tblr")
body <- cowplot::plot_grid(top_row, bottom_row, legend, ncol = 1,
                          rel_heights = c(1, 1, 0.18))
figure8 <- cowplot::ggdraw(body) +
  cowplot::draw_label("A", x = 0.008, y = 0.988, hjust = 0, vjust = 1,
                      fontface = "bold", size = 13, fontfamily = "Arial") +
  cowplot::draw_label("B", x = 0.008, y = 0.53, hjust = 0, vjust = 1,
                      fontface = "bold", size = 13, fontfamily = "Arial")

# 5. One figure, two journal-ready formats ------------------------------------
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
pdf_path <- file.path(output_dir,
  "Figure8_locked_XGBoost_calibration_DCA_three_cohorts.pdf")
tiff_path <- file.path(output_dir,
  "Figure8_locked_XGBoost_calibration_DCA_three_cohorts_600dpi.tiff")

if (capabilities("cairo")) {
  ggsave(pdf_path, figure8, device = grDevices::cairo_pdf,
         width = 7.1, height = 6.5, units = "in", bg = "white")
} else {
  ggsave(pdf_path, figure8, device = "pdf", useDingbats = FALSE,
         width = 7.1, height = 6.5, units = "in", bg = "white")
}
ggsave(tiff_path, figure8, device = "tiff", compression = "lzw",
       width = 7.1, height = 6.5, units = "in", dpi = 600, bg = "white")

# Manuscript legend (text only, not rendered inside the image):
# Figure 8. Calibration and decision-curve analysis of the locked XGBoost
# model for composite ESWL success. (A) Calibration in the training (apparent),
# internal test, and external validation cohorts. Red lines and shaded bands
# show logistic calibration estimates and pointwise 95% CIs; blue points and
# error bars show observed proportions with exact binomial 95% CIs. Training
# and internal sets use quintiles; the external set uses tertiles. Rug marks
# show predicted probabilities; the dashed diagonal is perfect calibration.
# The external set contained 13 failures, so estimates are uncertain.
# (B) DCA for XGBoost, treat all, and treat none across thresholds 0.10-0.90.
# Predictions came from the original locked model without recalibration.
