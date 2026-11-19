# ============================================================================
# XGBoost: composite first-session ESWL success, three-cohort validation
# Uses the uploaded model bundle, three-cohort RData and FROZEN OOF threshold CSV.
# No model fitting, tuning, recalibration or threshold re-estimation.
# success Yes -> 1 (positive); No -> 0. All metrics use P(success = 1).
# Training results are apparent/development results, NOT out-of-fold estimates.
# Prefer running in the SAME R/package environment that saved the fitted model.
# ============================================================================

# 1. Settings: edit paths, or select the files when prompted --------------------
input_dir <- "."   
model_file <- file.path(input_dir, "output/model_benchmark_9_CV_FORMAL/13_final_models/00_ESWL_all_9_best_CV_refit_models.rds")  
data_file  <- file.path(input_dir, "data/复合碎石成功的三个队列.RData")
threshold_file <- file.path(input_dir, "output/model_benchmark_9_CV_FORMAL/04_development_OOF_Youden_thresholds.csv")     
if (!file.exists(model_file)) model_file <- file.choose() # select model bundle .rds
if (!file.exists(data_file))  data_file  <- file.choose() # select three-cohort .RData
if (!file.exists(threshold_file)) threshold_file <- file.choose() # select frozen threshold .csv
output_dir <- file.path("output", "Table3")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# XGBoost's frozen development-OOF threshold in the supplied CSV:
# 0.7307063937187195. Read its full precision below; never re-optimize it.
B <- 2000L
seed <- 20260904L
font_family <- "sans"    # portable; change only to a font installed on your PC
save_tiff <- TRUE        # 600 dpi, LZW compression; PDF always saved
calibration_groups <- 5L
# Illustrative DCA: act/select ESWL when P(success) >= threshold.
# A clinical interpretation requires this action and utility trade-off to be valid.
dca_thresholds <- seq(0.05, 0.95, by = 0.005)

pkgs <- c("workflows", "parsnip", "recipes", "hardhat", "xgboost", "pROC",
          "dplyr", "tidyr", "ggplot2", "scales")
missing_pkgs <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs)) stop(
  "Install missing packages in your original modelling environment: ",
  paste(missing_pkgs, collapse = ", "),
  "\nDo not upgrade xgboost blindly: RDS models can be version-sensitive.")
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(ggplot2)
  library(workflows); library(parsnip); library(recipes); library(xgboost)
})
set.seed(seed)
write_csv <- function(x, filename) utils::write.csv(
  x, file.path(output_dir, filename), row.names = FALSE, na = "", fileEncoding = "UTF-8")

# 2. Load model and data in an isolated environment -----------------------------
model_object <- readRDS(model_file)
# The newly supplied RDS stores the fitted workflows under $models.
# Also accepts the previously supplied standalone XGBoost workflow.
if (inherits(model_object, "workflow")) {
  wf <- model_object
} else if (is.list(model_object) && "models" %in% names(model_object) &&
           "XGBoost" %in% names(model_object$models)) {
  wf <- model_object$models[["XGBoost"]]
} else {
  stop("Select the supplied nine-model RDS (models$XGBoost) or standalone XGBoost workflow.")
}
if (!inherits(wf, "workflow")) stop("XGBoost is not a fitted workflow.")

# Read the previously frozen threshold BEFORE accessing cohort outcomes.
threshold_table <- read.csv(threshold_file, stringsAsFactors = FALSE,
                            check.names = FALSE, fileEncoding = "UTF-8-BOM")
required_threshold_columns <- c("Model", "OOF_Youden_threshold")
if (!all(required_threshold_columns %in% names(threshold_table)))
  stop("Threshold CSV must contain Model and OOF_Youden_threshold.")
threshold_row <- which(trimws(as.character(threshold_table$Model)) == "XGBoost")
if (length(threshold_row) != 1L) stop("Threshold CSV must have exactly one XGBoost row.")
threshold <- suppressWarnings(as.numeric(threshold_table$OOF_Youden_threshold[threshold_row]))
if (length(threshold) != 1L || !is.finite(threshold) || threshold <= 0 || threshold >= 1)
  stop("The frozen XGBoost threshold must be strictly between 0 and 1.")
threshold_source <- paste0("Previously frozen development OOF Youden threshold; ", basename(threshold_file))
message(sprintf("Frozen XGBoost threshold: %.16f. No threshold re-estimation.", threshold))
model_fit <- workflows::extract_fit_parsnip(wf)
engine <- workflows::extract_fit_engine(wf)
if (!identical(as.character(model_fit$lvl), c("1", "0")))
  stop("Unexpected model class levels. This script expects levels c('1','0').")
# In this saved parsnip model, the first factor level is the engine event.
eng_event <- model_fit$spec$eng_args$event_level
if (!is.null(eng_event) && !identical(rlang::eval_tidy(eng_event), "first"))
  stop("Engine event_level differs from the inspected model.")

# The uploaded RDS has best_ntreelimit = 68, niter = 93.
# Explicitly use the saved best iteration, not all 93 trees retained after stopping.
# R's iterationrange is 1-based, with an exclusive upper bound.
best_rounds <- if (is.list(engine)) engine$best_ntreelimit else NULL
if (is.null(best_rounds)) {
  # This fallback is specific to the inspected attachment, not a new model choice.
  best_rounds <- 68L
}
best_rounds <- as.integer(best_rounds)
if (length(best_rounds) != 1L || is.na(best_rounds) || best_rounds != 68L)
  stop("The model's best iteration differs from the inspected 68-round model.")

e <- new.env(parent = emptyenv())
load(data_file, envir = e)
object_names <- c("training_data", "test_data", "external_data")
if (!all(vapply(object_names, exists, logical(1), envir = e, inherits = FALSE)))
  stop("RData must contain training_data, test_data and external_data.")
cohort_names <- c("Training", "Internal test", "External validation")
cohorts <- setNames(lapply(object_names, get, envir = e), cohort_names)
name_map <- c(
  "Stone volume" = "Stone_volume",
  "Mean CT attenuation" = "Mean_CT_attenuation",
  "Ureteral wall thickness" = "Ureteral_wall_thickness",
  "hydronephrosis" = "Hydronephrosis_measure",
  "Subcutaneous fat thickness" = "Subcutaneous_fat_thickness",
  "Stone location" = "Stone_location"
)
location_levels <- c("Distal ureter", "Mid ureter", "Proximal ureter")
prepare_data <- function(d) {
  d <- as.data.frame(d); names(d) <- trimws(names(d))
  if (!all(c(names(name_map), "success") %in% names(d)))
    stop("Missing original data columns: ", paste(setdiff(c(names(name_map), "success"), names(d)), collapse = ", "))
  names(d)[match(names(name_map), names(d))] <- unname(name_map)
  yy <- trimws(as.character(d$success))
  if (anyNA(yy) || !all(yy %in% c("Yes", "No"))) stop("success must contain only Yes/No.")
  d$truth01 <- as.integer(yy == "Yes")
  d$success <- factor(ifelse(d$truth01 == 1L, "1", "0"), levels = c("1", "0"))
  z <- trimws(as.character(d$Stone_location))
  if (anyNA(z) || !all(z %in% location_levels)) stop("Unexpected Stone location category.")
  d$Stone_location <- factor(z, levels = location_levels)
  for (v in setdiff(unname(name_map), "Stone_location")) {
    d[[v]] <- suppressWarnings(as.numeric(as.character(d[[v]])))
    if (any(!is.finite(d[[v]]))) stop("Missing/non-numeric predictor: ", v)
  }
  # Local row index meets the recipe's case_id role; NEVER used as a predictor.
  # This is not a patient identifier and cannot detect cross-cohort patient overlap.
  d$Case_ID <- seq_len(nrow(d))
  if (length(unique(d$truth01)) != 2L) stop("Each cohort must contain both outcomes.")
  d
}
cohorts <- lapply(cohorts, prepare_data)
if (!identical(as.integer(vapply(cohorts, nrow, integer(1))), c(621L, 208L, 147L)))
  stop("Cohort sizes differ from the supplied 621/208/147 data.")

# Check that the uploaded training rows/outcomes match those saved in the model.
mold <- workflows::extract_mold(wf)
forged_train <- hardhat::forge(cohorts[[1]], blueprint = mold$blueprint)$predictors
if (!isTRUE(all.equal(as.matrix(forged_train), as.matrix(mold$predictors),
                      check.attributes = FALSE, tolerance = 1e-10)))
  stop("Training predictors do not match the model's saved development data.")
if (!identical(as.character(cohorts[[1]]$success), as.character(mold$outcomes$success)))
  stop("Training outcomes do not match the saved model outcomes.")

# Workflow applies its SAVED recipe. type=raw permits an explicit iteration range.
# For this binary parsnip model raw probabilities refer to first level '1'.
predictions <- bind_rows(lapply(seq_along(cohorts), function(i) {
  d <- cohorts[[i]]
  p <- tryCatch(as.numeric(predict(wf, new_data = d, type = "raw",
                   opts = list(iterationrange = c(1L, best_rounds + 1L)))),
                error = function(err) stop("Prediction failed. Use original xgboost/parsnip versions.\n", conditionMessage(err)))
  if (length(p) != nrow(d) || any(!is.finite(p)) || any(p < 0 | p > 1))
    stop("Invalid probability predictions.")
  data.frame(Cohort = cohort_names[i], Row_ID = seq_len(nrow(d)),
             truth01 = d$truth01, probability = p)
}))
predictions$Cohort <- factor(predictions$Cohort, levels = cohort_names)
cohort_summary <- predictions |>
  group_by(Cohort) |>
  summarise(n = n(), Success = sum(truth01), Failure = sum(1L - truth01),
            Observed_success = mean(truth01), Mean_predicted = mean(probability), .groups = "drop")
write_csv(cohort_summary, "00_Cohort_summary.csv")

# 3. Apply the PREVIOUSLY FROZEN threshold unchanged to every cohort ------------
# No training ROC threshold search, fallback, or threshold re-estimation exists.
# CSV value is used at full precision; rounding is for display only.
predictions$predicted01 <- as.integer(predictions$probability >= threshold)
write_csv(predictions, "01_Predictions.csv")
message(sprintf("Applied frozen success threshold: %.16f (%s)", threshold, threshold_source))

# 4. Statistics and CIs --------------------------------------------------------
# AUC: DeLong; accuracy/sensitivity/specificity/PPV/NPV: exact binomial.
# Calibration intercept = calibration-in-the-large (slope fixed to 1).
# Calibration slope = coefficient of logit(p) with a freely estimated intercept.
# Brier/calibration CIs: ordinary patient bootstrap, 2000 draws, percentile CI.
# Fixed predictions and threshold: conditional CIs, no model refitting or selection.
# CIs condition on the frozen model and threshold; training CIs do NOT correct optimism.
clip_p <- function(p) pmin(pmax(p, 1e-6), 1 - 1e-6)
safe_glm <- function(formula, data) {
  f <- suppressWarnings(tryCatch(glm(formula, data = data, family = binomial(),
                                     control = glm.control(maxit = 100)), error = function(e) NULL))
  if (is.null(f) || !isTRUE(f$converged) || isTRUE(f$boundary) || any(!is.finite(coef(f)))) return(NULL)
  f
}
cal_metrics <- function(y, p) {
  dd <- data.frame(y = y, lp = qlogis(clip_p(p)))
  a <- safe_glm(y ~ 1 + offset(lp), dd)
  b <- safe_glm(y ~ lp, dd)
  c(Calibration_intercept = if (is.null(a)) NA_real_ else unname(coef(a)[1]),
    Calibration_slope = if (is.null(b)) NA_real_ else unname(coef(b)[2]),
    Joint_intercept = if (is.null(b)) NA_real_ else unname(coef(b)[1]),
    Brier_score = mean((p - y)^2))
}
exact_metric <- function(k, n) {
  if (n == 0) return(c(estimate = NA, lower = NA, upper = NA))
  c(estimate = k / n, lower = binom.test(k, n)$conf.int[1], upper = binom.test(k, n)$conf.int[2])
}
percentile_ci <- function(x) {
  # Do not silently discard many unstable/separated bootstrap fits.
  if (mean(is.finite(x)) < 0.95) return(c(NA_real_, NA_real_))
  unname(quantile(x[is.finite(x)], c(.025, .975), type = 7))
}
pr_points <- function(y, p) {
  dd <- data.frame(y, p) |> group_by(p) |>
    summarise(positive = sum(y), total = n(), .groups = "drop") |> arrange(desc(p)) |>
    mutate(tp = cumsum(positive), fp = cumsum(total - positive),
           Recall = tp / sum(y), Precision = tp / (tp + fp))
  dd <- bind_rows(data.frame(Recall = 0, Precision = 1), as.data.frame(dd[, c("Recall", "Precision")]))
  dd
}

results <- lapply(cohort_names, function(cohort) {
  message("Evaluating ", cohort, "; bootstrap B = ", B)
  d <- filter(predictions, Cohort == cohort); y <- d$truth01; p <- d$probability; n <- nrow(d)
  tp <- sum(y == 1 & d$predicted01 == 1); tn <- sum(y == 0 & d$predicted01 == 0)
  fp <- sum(y == 0 & d$predicted01 == 1); fn <- sum(y == 1 & d$predicted01 == 0)
  roc <- pROC::roc(y, p, levels = c(0, 1), direction = "<", quiet = TRUE)
  auc_ci <- as.numeric(pROC::ci.auc(roc, method = "delong"))
  counts <- list(Accuracy = c(tp + tn, n), Sensitivity = c(tp, tp + fn),
                 Specificity = c(tn, tn + fp), PPV = c(tp, tp + fp), NPV = c(tn, tn + fn))
  ratios <- t(vapply(counts, function(v) exact_metric(v[1], v[2]), numeric(3)))
  estimates <- cal_metrics(y, p)

  # Flexible calibration: logistic natural cubic spline of logit(p), df=3.
  # Fix spline basis/knots to the cohort for pointwise bootstrap intervals.
  lp <- qlogis(clip_p(p)); p_grid <- seq(min(p), max(p), length.out = 160)
  basis <- splines::ns(lp, df = 3)
  grid_basis <- predict(basis, qlogis(clip_p(p_grid)))
  smooth_dat <- data.frame(y = y, s1 = basis[,1], s2 = basis[,2], s3 = basis[,3])
  smooth_grid <- data.frame(s1 = grid_basis[,1], s2 = grid_basis[,2], s3 = grid_basis[,3])
  smooth_fit <- safe_glm(y ~ s1 + s2 + s3, smooth_dat)
  smooth_est <- if (is.null(smooth_fit)) rep(NA_real_, length(p_grid)) else
    as.numeric(predict(smooth_fit, newdata = smooth_grid, type = "response"))
  boot_metric <- matrix(NA_real_, B, length(estimates), dimnames = list(NULL, names(estimates)))
  boot_curve <- matrix(NA_real_, B, length(p_grid))
  for (b in seq_len(B)) {
    ix <- sample.int(n, n, replace = TRUE)
    boot_metric[b, "Brier_score"] <- mean((y[ix] - p[ix])^2)
    if (length(unique(y[ix])) == 2L) {
      boot_metric[b, ] <- cal_metrics(y[ix], p[ix])
      sf <- safe_glm(y ~ s1 + s2 + s3, smooth_dat[ix, , drop = FALSE])
      if (!is.null(sf)) boot_curve[b, ] <- as.numeric(predict(sf, newdata = smooth_grid, type = "response"))
    }
  }
  metric_ci <- t(apply(boot_metric, 2, percentile_ci))
  curve_ci <- t(apply(boot_curve, 2, percentile_ci))
  audit <- data.frame(Cohort = cohort, Statistic = c(colnames(boot_metric), "Calibration_curve"),
                      B_requested = B, B_valid = c(colSums(is.finite(boot_metric)),
                                                   sum(apply(boot_curve, 1, function(z) all(is.finite(z))))))
  audit$Valid_fraction <- audit$B_valid / B
  if (any(audit$Valid_fraction < .95))
    warning(cohort, ": fewer than 95% valid bootstrap fits; affected CIs suppressed. See audit CSV.")
  long <- bind_rows(
    data.frame(Metric = "AUC", Estimate = as.numeric(pROC::auc(roc)), Lower = auc_ci[1], Upper = auc_ci[3]),
    data.frame(Metric = rownames(ratios), Estimate = ratios[,1], Lower = ratios[,2], Upper = ratios[,3]),
    data.frame(Metric = names(estimates), Estimate = as.numeric(estimates), Lower = metric_ci[,1], Upper = metric_ci[,2]))
  long$Cohort <- cohort
  pr <- pr_points(y, p)
  ap <- sum(diff(pr$Recall) * pr$Precision[-1]) # average precision, ties grouped
  denom <- sqrt(as.double((tp+fp)*(tp+fn)) * as.double((tn+fp)*(tn+fn)))
  extra <- data.frame(Cohort = cohort, F1 = 2*tp/(2*tp+fp+fn),
                       Balanced_accuracy = (tp/(tp+fn) + tn/(tn+fp))/2,
                       MCC = if (denom == 0) NA_real_ else (tp*tn-fp*fn)/denom,
                       Average_precision = ap,
                       Brier_skill_score = 1 - mean((y-p)^2)/(mean(y)*(1-mean(y))),
                       Log_loss = -mean(y*log(clip_p(p)) + (1-y)*log1p(-clip_p(p))))
  # Quantile bins: tied probabilities stay in the same bin.
  breaks <- unique(as.numeric(quantile(p, seq(0, 1, length.out = calibration_groups + 1L))))
  bins <- data.frame(y, p, bin = cut(p, breaks, include.lowest = TRUE)) |>
    group_by(bin) |> summarise(Predicted = mean(p), Observed = mean(y), n = n(), .groups = "drop")
  list(long = long, roc = data.frame(FPR = 1-roc$specificities, TPR = roc$sensitivities, Cohort = cohort),
       pr = transform(pr, Cohort = cohort), extra = extra, audit = audit,
       cm = data.frame(Cohort = cohort, TN = tn, FP = fp, FN = fn, TP = tp),
       curve = data.frame(Cohort = cohort, Predicted = p_grid, Observed = smooth_est,
                          Lower = curve_ci[,1], Upper = curve_ci[,2]),
       bins = transform(bins, Cohort = cohort))
})
long_results <- bind_rows(lapply(results, `[[`, "long"))
cm <- bind_rows(lapply(results, `[[`, "cm"))
extra <- bind_rows(lapply(results, `[[`, "extra"))
write_csv(long_results, "02_Performance_estimates_95CI.csv")
write_csv(extra, "03_Supplementary_metrics.csv")
write_csv(bind_rows(lapply(results, `[[`, "audit")), "04_Bootstrap_audit.csv")
write_csv(cm, "05_Confusion_counts.csv")

# 5. Table 3: template order, dynamically calculated sample sizes ---------------
metric_order <- c("AUC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV",
                  "Calibration_intercept", "Calibration_slope", "Brier_score")
metric_labels <- c("AUC (95% CI)", "Accuracy (95% CI)", "Sensitivity (95% CI)",
                   "Specificity (95% CI)", "PPV (95% CI)", "NPV (95% CI)",
                   "Calibration intercept (95% CI)", "Calibration slope (95% CI)", "Brier score (95% CI)")
fmt <- function(a, l, u) ifelse(!is.finite(a), "NE", ifelse(!is.finite(l) | !is.finite(u),
             sprintf("%.3f (CI not estimable)", a), sprintf("%.3f (%.3f-%.3f)", a, l, u)))
table3 <- data.frame(`Performance metric` = c(metric_labels, "TN / FP / FN / TP"), check.names = FALSE)
for (cohort in cohort_names) {
  v <- filter(long_results, Cohort == cohort)
  v <- v[match(metric_order, v$Metric), ]
  ct <- filter(cm, Cohort == cohort)
  col <- sprintf("%s cohort (n=%d)", cohort, sum(predictions$Cohort == cohort))
  table3[[col]] <- c(fmt(v$Estimate, v$Lower, v$Upper), paste(unlist(ct[c("TN", "FP", "FN", "TP")]), collapse = " / "))
}
write_csv(table3, "Table3_XGBoost_Three_Cohorts.csv")
footnote <- paste0(
  "Values are estimates (95% CIs), unless otherwise indicated. Composite first-session ESWL success (Yes/1) is the positive event. ",
  "All classification metrics use a common threshold of ", sprintf("%.16f", threshold), "; source: ", threshold_source, ". ",
  "The previously frozen development OOF threshold was read from the supplied CSV and applied unchanged to all three cohorts. No threshold was re-estimated in this evaluation. No model tuning or recalibration was performed. ",
  "AUC CIs: DeLong; accuracy, sensitivity, specificity, PPV and NPV CIs: exact binomial. ",
  "Calibration and Brier CIs: ", B, " ordinary patient bootstrap draws with fixed predictions. ",
  "Calibration intercept is estimated with logit(p) as an offset (slope fixed at 1); calibration slope uses a free intercept. ",
  "The joint intercept is provided separately. CIs are suppressed when fewer than 95% of bootstrap estimates are valid. ",
  "Training results are apparent/development performance, not OOF or optimism-corrected performance. ",
  "The saved model used a 20% development holdout for early stopping; predictions use its saved best 68 boosting rounds. ",
  "The label 'training cohort' describes all 621 development observations, not 621 observations all used for tree fitting. ",
  "PPV, positive predictive value; NPV, negative predictive value; TN/FP/FN/TP, true negative/false positive/false negative/true positive; NE, not estimable.")
html_escape <- function(x) {
  x <- gsub("&", "&amp;", x, fixed = TRUE); x <- gsub("<", "&lt;", x, fixed = TRUE)
  gsub(">", "&gt;", x, fixed = TRUE)
}
html <- c('<!doctype html><html><head><meta charset="utf-8"><style>',
  'body{font-family:"Times New Roman",serif;max-width:1100px;margin:36px auto;color:#111}',
  'h2{font-size:18px;line-height:1.35}table{border-collapse:collapse;width:100%;font-size:15px;border-top:2px solid #111;border-bottom:2px solid #111}',
  'th{border-bottom:1px solid #111;padding:10px 8px}td{padding:8px;text-align:center}th:first-child,td:first-child{text-align:left}p{font-size:13px;line-height:1.45}',
  '</style></head><body><h2>Table 3. Performance of the fixed XGBoost model across the training, internal test and external validation cohorts</h2><table><thead><tr>',
  paste0('<th>', html_escape(names(table3)), '</th>', collapse = ''), '</tr></thead><tbody>',
  apply(table3, 1, function(v) paste0('<tr>', paste0('<td>', html_escape(v), '</td>', collapse = ''), '</tr>')),
  '</tbody></table>', paste0('<p>', html_escape(footnote), '</p></body></html>'))
writeLines(html, file.path(output_dir, "Table3_XGBoost_Three_Cohorts.html"), useBytes = TRUE)

# 6. Publication-oriented figures: consistent palette, typography and scales ---
cols <- c("Training" = "#0072B2", "Internal test" = "#D55E00", "External validation" = "#009E73")
lt <- c("Training" = "solid", "Internal test" = "longdash", "External validation" = "dotdash")
cohort_factor <- function(d) { d$Cohort <- factor(d$Cohort, levels = cohort_names); d }
base_theme <- theme_classic(base_size = 10, base_family = font_family) +
  theme(axis.title = element_text(size = 10), axis.text = element_text(color = "#222222"),
        axis.line = element_line(linewidth = .35), axis.ticks = element_line(linewidth = .3),
        strip.background = element_blank(), strip.text = element_text(face = "bold", size = 10),
        legend.position = "bottom", legend.title = element_blank(),
        legend.text = element_text(size = 9), legend.key.width = grid::unit(1.3, "cm"),
        plot.title = element_text(face = "bold", size = 12),
        plot.subtitle = element_text(size = 9, color = "#444444"),
        plot.caption = element_text(size = 8, hjust = 0, color = "#444444"),
        plot.margin = margin(8, 10, 8, 8), panel.spacing = grid::unit(1.1, "lines"))
theme_set(base_theme)
export_plot <- function(p, name, width, height) {
  pdf_device <- if (capabilities("cairo")) grDevices::cairo_pdf else grDevices::pdf
  ggsave(file.path(output_dir, paste0(name, ".pdf")), p, device = pdf_device,
         width = width, height = height, units = "in", bg = "white")
  ggsave(file.path(output_dir, paste0(name, ".png")), p,
         width = width, height = height, units = "in", dpi = 300, bg = "white")
  if (save_tiff) ggsave(file.path(output_dir, paste0(name, ".tiff")), p, device = "tiff",
                       width = width, height = height, units = "in", dpi = 600,
                       compression = "lzw", bg = "white")
}
roc_data <- cohort_factor(bind_rows(lapply(results, `[[`, "roc")))
auc_rows <- filter(long_results, Metric == "AUC")
auc_rows <- auc_rows[match(cohort_names, auc_rows$Cohort), ]
roc_labels <- setNames(paste0(cohort_names, ": ", fmt(auc_rows$Estimate, auc_rows$Lower, auc_rows$Upper)), cohort_names)
p_roc <- ggplot(roc_data, aes(FPR, TPR, color = Cohort, linetype = Cohort)) +
  geom_abline(slope = 1, intercept = 0, color = "#AAAAAA", linewidth = .4, linetype = "dashed") +
  geom_path(linewidth = .85) +
  scale_color_manual(values = cols, breaks = cohort_names, labels = roc_labels) +
  scale_linetype_manual(values = lt, breaks = cohort_names, labels = roc_labels) +
  scale_x_continuous(breaks = seq(0,1,.2), expand = expansion(mult = .01)) +
  scale_y_continuous(breaks = seq(0,1,.2), expand = expansion(mult = .01)) + coord_equal() +
  labs(x = "1 - Specificity", y = "Sensitivity", title = "Discrimination",
       subtitle = "XGBoost | Composite ESWL success", caption = "Legend: AUC (95% CI, DeLong). Training performance is apparent.") +
  guides(color = guide_legend(ncol = 1), linetype = guide_legend(ncol = 1))
export_plot(p_roc, "Diagnostic_ROC", 5.5, 6.1)

curve_data <- cohort_factor(bind_rows(lapply(results, `[[`, "curve")))
bin_data <- cohort_factor(bind_rows(lapply(results, `[[`, "bins")))
write_csv(curve_data, "06_Calibration_curve_data.csv")
write_csv(bin_data, "07_Calibration_group_data.csv")
p_cal <- ggplot(curve_data, aes(Predicted, Observed, color = Cohort, fill = Cohort)) +
  geom_abline(slope = 1, intercept = 0, color = "#999999", linewidth = .4, linetype = "dashed") +
  geom_ribbon(aes(ymin = Lower, ymax = Upper), alpha = .16, color = NA, na.rm = TRUE) +
  geom_line(linewidth = .8, na.rm = TRUE) +
  geom_point(data = bin_data, shape = 21, fill = "white", size = 2, stroke = .65) +
  facet_wrap(~Cohort, nrow = 1) + scale_color_manual(values = cols) + scale_fill_manual(values = cols) +
  scale_x_continuous(breaks = seq(0,1,.25), limits = c(0,1), expand = expansion(mult = .015)) +
  scale_y_continuous(breaks = seq(0,1,.25), limits = c(0,1), expand = expansion(mult = .015)) + coord_equal() +
  labs(x = "Predicted success probability", y = "Observed success probability", title = "Calibration",
       caption = paste0("Logistic natural cubic spline; shaded bands: pointwise 95% bootstrap CIs.\nOpen circles: quantile groups (ties retained); dashed line: ideal calibration. No probabilities were recalibrated.")) +
  theme(legend.position = "none")
export_plot(p_cal, "Diagnostic_Calibration", 9, 3.8)

# DCA displays net benefit over an illustrative threshold grid; this does NOT
# select, optimize or change the frozen classification threshold used in Table 3.
# It evaluates an assumed action for predicted success, not a causal ESWL benefit.
dca_data <- bind_rows(lapply(cohort_names, function(cohort) {
  d <- filter(predictions, Cohort == cohort); n <- nrow(d); prevalence <- mean(d$truth01)
  nb <- vapply(dca_thresholds, function(t) {
    selected <- d$probability >= t
    sum(selected & d$truth01 == 1)/n - sum(selected & d$truth01 == 0)/n*t/(1-t)
  }, numeric(1))
  data.frame(Cohort = cohort, Threshold = rep(dca_thresholds, 3),
             Strategy = rep(c("XGBoost", "Act for all", "Act for none"), each = length(dca_thresholds)),
             Net_benefit = c(nb, prevalence - (1-prevalence)*dca_thresholds/(1-dca_thresholds),
                             rep(0, length(dca_thresholds))))
})) |> cohort_factor()
write_csv(dca_data, "08_Decision_curve_data.csv")
p_dca <- ggplot(dca_data, aes(Threshold, Net_benefit, group = Strategy)) +
  geom_line(data = filter(dca_data, Strategy != "XGBoost"), aes(linetype = Strategy), color = "#777777", linewidth = .55) +
  geom_line(data = filter(dca_data, Strategy == "XGBoost"), aes(color = Cohort, linetype = Strategy), linewidth = .85) +
  facet_wrap(~Cohort, nrow = 1) + scale_color_manual(values = cols, guide = "none") +
  scale_linetype_manual(values = c("XGBoost" = "solid", "Act for all" = "longdash", "Act for none" = "dotted"),
                        breaks = c("XGBoost", "Act for all", "Act for none")) +
  scale_x_continuous(breaks = c(.1,.3,.5,.7,.9)) +
  coord_cartesian(ylim = c(-.1,1)) +
  labs(x = "Threshold probability of success", y = "Net benefit", title = "Exploratory decision curve analysis",
       caption = "Assumed action: select ESWL when predicted success exceeds the threshold; clinical utilities must support this rule.\nDisplay truncated below -0.10; complete values are exported. Curves are point estimates, not evidence of causal treatment benefit.")
export_plot(p_dca, "Diagnostic_Decision_Curves", 9, 4.2)

pr_data <- cohort_factor(bind_rows(lapply(results, `[[`, "pr")))
pr_baselines <- cohort_factor(as.data.frame(cohort_summary))
p_pr <- ggplot(pr_data, aes(Recall, Precision, color = Cohort)) +
  geom_hline(data = pr_baselines, aes(yintercept = Observed_success), color = "#999999", linewidth = .4, linetype = "dashed") +
  geom_step(direction = "vh", linewidth = .85) + facet_wrap(~Cohort, nrow = 1) +
  scale_color_manual(values = cols) +
  scale_x_continuous(breaks = seq(0,1,.25), limits = c(0,1), expand = expansion(mult = .015)) +
  scale_y_continuous(breaks = seq(0,1,.25), limits = c(0,1), expand = expansion(mult = .015)) + coord_equal() +
  labs(x = "Recall (sensitivity)", y = "Precision (PPV)", title = "Precision-recall curves",
       caption = "Dashed lines: cohort-specific success prevalence. Average precision is reported in supplementary metrics.") +
  theme(legend.position = "none")
export_plot(p_pr, "Diagnostic_Precision_Recall", 9, 3.7)

cm_long <- cm |> pivot_longer(c(TN,FP,FN,TP), names_to = "Cell", values_to = "Count") |>
  mutate(Observed = ifelse(Cell %in% c("TN","FP"), "Failure", "Success"),
         Predicted = ifelse(Cell %in% c("TN","FN"), "Failure", "Success")) |>
  group_by(Cohort, Observed) |> mutate(Proportion = Count/sum(Count)) |> ungroup() |>
  mutate(Label = sprintf("%s = %d\n%.1f%%", Cell, Count, 100*Proportion),
         Observed = factor(Observed, levels = c("Success", "Failure")),
         Predicted = factor(Predicted, levels = c("Failure", "Success"))) |> cohort_factor()
p_cm <- ggplot(cm_long, aes(Predicted, Observed, fill = Proportion)) +
  geom_tile(color = "white", linewidth = .8) +
  geom_text(aes(label = Label, color = Proportion > .55), size = 3.4, lineheight = 1.15) +
  scale_color_manual(values = c("FALSE" = "#202020", "TRUE" = "white"), guide = "none") +
  scale_fill_gradient(low = "#F0F4F7", high = "#17628A", limits = c(0,1), labels = scales::percent, name = "Within outcome") +
  facet_wrap(~Cohort, nrow = 1) + coord_equal() +
  labs(x = "Predicted outcome", y = "Observed outcome", title = "Classification at the common threshold",
       subtitle = sprintf("P(success) >= %.6f", threshold), caption = "Percentages are calculated within each observed outcome (row), not within the whole cohort.") +
  theme(axis.line = element_blank(), axis.ticks = element_blank())
export_plot(p_cm, "Diagnostic_Confusion_Matrices", 9, 3.9)

# 7. Audit and reproducibility -------------------------------------------------
writeLines(c(footnote,
  "", "Interpretation notes:",
  "1. High AUC does not imply good calibration; inspect the full calibration curve and both calibration parameters.",
  "2. External specificity is estimated among 13 composite failures. See the performance table for the exact CI.",
  "3. PPV/NPV, accuracy, PR and Brier score depend on event prevalence and case mix; avoid simple cohort ranking.",
  "4. No subject IDs were supplied, so patient independence and overlap cannot be checked from these files.",
  "5. Numeric agreement for hydronephrosis is verified in training, but common units/measurement definitions across cohorts require source-data verification.",
  "6. These data contain a precomputed success flag. The components and time window defining composite first-session success cannot be reconstructed from the seven columns.",
  "7. No SHAP or feature-importance plots: those explain predictions and are not performance validation.",
  "8. Table 3 and confusion matrices use only the frozen OOF threshold. The DCA threshold grid is descriptive and never changes this threshold.",
  "", "Methods references:",
  "https://pmc.ncbi.nlm.nih.gov/articles/PMC6912996/",
  "https://pmc.ncbi.nlm.nih.gov/articles/PMC10788734/",
  "https://workflows.tidymodels.org/reference/predict-workflow.html",
  "https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html"),
  file.path(output_dir, "Methods_and_interpretation.txt"), useBytes = TRUE)
write_csv(data.frame(Item = c("Positive_event", "Probability", "Best_rounds", "Threshold", "Threshold_source", "Bootstrap_draws", "Seed", "Training_evaluation"),
                     Value = c("Yes -> 1", "Saved parsnip engine first level: 1", best_rounds,
                               sprintf("%.16f", threshold), threshold_source, B, seed, "Apparent/development")),
          "09_Run_configuration.csv")
write_csv(data.frame(File = basename(c(model_file, data_file, threshold_file)), MD5 = unname(tools::md5sum(c(model_file, data_file, threshold_file)))),
          "10_Input_checksums.csv")
capture.output(sessionInfo(), file = file.path(output_dir, "sessionInfo.txt"))
saveRDS(list(table3 = table3, estimates = long_results, supplementary = extra, threshold = threshold,
             threshold_source = threshold_source, predictions = predictions, results = results,
             dca = dca_data, best_rounds = best_rounds),
        file.path(output_dir, "XGBoost_Three_Cohorts_Evaluation.rds"))
print(table3, row.names = FALSE)
message("Finished. Outputs: ", normalizePath(output_dir, winslash = "/"))
