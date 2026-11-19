# ==============================================================================
# Corrected threshold analysis: MeCT and subcutaneous fat thickness (SFT)
# Breakpoint: two-stage maximum-likelihood grid search
# Global slope-change test: Davies test (not the minimum grid-search LRT P)
# Output: one CSV table; an additional Word table if officer/flextable exist
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tibble)
})

if (!requireNamespace("segmented", quietly = TRUE)) {
  stop("Please install 'segmented': install.packages('segmented', type='binary')")
}

# ---- Only edit this block -----------------------------------------------------
data_file <- "data/831ESWL.csv"
output_dir <- "output/Table2"

search_prob <- c(0.10, 0.90)
davies_points <- 100

# Same two-stage method, with scale-appropriate step sizes
MeCT_coarse <- 5.0       # HU
MeCT_refine <- 0.1       # HU
MeCT_half_width <- 10.0  # HU around the best coarse value

SFT_coarse <- 0.5        # mm
SFT_refine <- 0.01       # mm
SFT_half_width <- 1.0    # mm around the best coarse value
# ------------------------------------------------------------------------------

raw <- read.csv(data_file, stringsAsFactors = FALSE, check.names = FALSE)

outcome_raw <- trimws(tolower(as.character(raw[["Stone success"]])))
dat <- tibble(
  success = case_when(
    outcome_raw %in% c("yes", "1", "success", "successful") ~ 1,
    outcome_raw %in% c("no", "0", "failure", "failed") ~ 0,
    TRUE ~ NA_real_
  ),
  MeCT = as.numeric(raw[["Mean CT attenuation"]]),
  SFT  = as.numeric(raw[["Subcutaneous fat thickness"]]),
  SV   = as.numeric(raw[["Stone volume"]]),
  UWT  = as.numeric(raw[["Ureteral wall thickness"]]),
  HYD  = as.numeric(raw[["Degree of hydronephrosis"]]),
  LOC  = factor(trimws(as.character(raw[["Stone location"]])))
) |>
  tidyr::drop_na()

if (nrow(dat) == 0L || length(unique(dat$success)) != 2L) {
  stop("No usable complete cases, or the outcome does not contain both classes.")
}

or_ci_p <- function(beta, se, increment) {
  z <- beta / se
  c(
    OR = exp(beta * increment),
    low = exp((beta - qnorm(0.975) * se) * increment),
    high = exp((beta + qnorm(0.975) * se) * increment),
    p = 2 * pnorm(-abs(z))
  )
}

fmt_p <- function(p) {
  if (is.na(p)) return("NA")
  if (p < 0.001) "<0.001" else sprintf("%.3f", p)
}

fmt_or <- function(x) {
  sprintf("%.3f (%.3f-%.3f)", x["OR"], x["low"], x["high"])
}

threshold_model <- function(data, exposure, covariates, increment,
                            coarse, refine, half_width,
                            probs = search_prob,
                            n_davies = davies_points) {

  dd <- data |>
    transmute(
      success,
      x = .data[[exposure]],
      across(all_of(covariates))
    )

  rhs <- paste(c("x", covariates), collapse = " + ")
  f_linear <- as.formula(paste("success ~", rhs))
  f_piece <- as.formula(
    paste("success ~ x + hinge +", paste(covariates, collapse = " + "))
  )

  m_linear <- glm(f_linear, data = dd, family = binomial())
  if (!isTRUE(m_linear$converged)) stop(exposure, ": linear model did not converge.")

  limits <- as.numeric(quantile(dd$x, probs = probs, na.rm = TRUE))

  loglik_at <- function(breakpoint) {
    d2 <- mutate(dd, hinge = pmax(x - breakpoint, 0))
    fit <- suppressWarnings(glm(f_piece, data = d2, family = binomial()))
    if (!isTRUE(fit$converged)) return(NA_real_)
    as.numeric(logLik(fit))
  }

  best_from_grid <- function(grid) {
    ll <- vapply(grid, loglik_at, numeric(1))
    if (!any(is.finite(ll))) stop(exposure, ": no candidate model converged.")
    grid[which.max(ll)]
  }

  coarse_grid <- seq(
    ceiling(limits[1] / coarse) * coarse,
    floor(limits[2] / coarse) * coarse,
    by = coarse
  )
  if (length(coarse_grid) < 2L) stop(exposure, ": coarse grid is too short.")
  best_coarse <- best_from_grid(coarse_grid)

  refine_grid <- seq(
    max(limits[1], best_coarse - half_width),
    min(limits[2], best_coarse + half_width),
    by = refine
  )
  breakpoint <- best_from_grid(refine_grid)

  d_final <- mutate(dd, hinge = pmax(x - breakpoint, 0))
  m_piece <- glm(f_piece, data = d_final, family = binomial())
  if (!isTRUE(m_piece$converged)) stop(exposure, ": final piecewise model did not converge.")

  # Global test for a slope change somewhere in the prespecified search range.
  davies_values <- seq(limits[1], limits[2], length.out = n_davies)
  davies <- segmented::davies.test(
    m_linear,
    seg.Z = ~x,
    values = davies_values,
    type = "lrt",
    alternative = "two.sided"
  )

  linear <- coef(summary(m_linear))
  piece <- coef(summary(m_piece))
  V <- vcov(m_piece)

  effect_linear <- or_ci_p(
    linear["x", "Estimate"], linear["x", "Std. Error"], increment
  )
  effect_below <- or_ci_p(
    piece["x", "Estimate"], piece["x", "Std. Error"], increment
  )

  beta_above <- piece["x", "Estimate"] + piece["hinge", "Estimate"]
  se_above <- sqrt(
    V["x", "x"] + V["hinge", "hinge"] + 2 * V["x", "hinge"]
  )
  effect_above <- or_ci_p(beta_above, se_above, increment)

  list(
    breakpoint = breakpoint,
    linear = effect_linear,
    below = effect_below,
    above = effect_above,
    davies_p = unname(davies$p.value),
    n = nrow(dd),
    n_below = sum(dd$x < breakpoint),
    n_above = sum(dd$x >= breakpoint)
  )
}

# Each exposure is additionally adjusted for the other exposure.
mect <- threshold_model(
  dat,
  exposure = "MeCT",
  covariates = c("SV", "UWT", "HYD", "SFT", "LOC"),
  increment = 20,
  coarse = MeCT_coarse,
  refine = MeCT_refine,
  half_width = MeCT_half_width
)

sft <- threshold_model(
  dat,
  exposure = "SFT",
  covariates = c("SV", "UWT", "HYD", "MeCT", "LOC"),
  increment = 5,
  coarse = SFT_coarse,
  refine = SFT_refine,
  half_width = SFT_half_width
)

result_table <- tibble(
  Parameter = c(
    "Model 1: Standard linear regression (adjusted)",
    "Model 2: Two-piecewise linear regression (adjusted)",
    "Inflection point",
    "Below inflection point",
    "Above inflection point",
    "P for Davies test (slope change)"
  ),
  MeCT_OR = c(
    fmt_or(mect$linear), "-", sprintf("%.1f HU", mect$breakpoint),
    fmt_or(mect$below), fmt_or(mect$above), "-"
  ),
  MeCT_P = c(
    fmt_p(mect$linear["p"]), "-", "-",
    fmt_p(mect$below["p"]), fmt_p(mect$above["p"]),
    fmt_p(mect$davies_p)
  ),
  SFT_OR = c(
    fmt_or(sft$linear), "-", sprintf("%.2f mm", sft$breakpoint),
    fmt_or(sft$below), fmt_or(sft$above), "-"
  ),
  SFT_P = c(
    fmt_p(sft$linear["p"]), "-", "-",
    fmt_p(sft$below["p"]), fmt_p(sft$above["p"]),
    fmt_p(sft$davies_p)
  )
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# The CSV result is always produced and does not require Word packages.
csv_file <- file.path(output_dir, "Table2_threshold_effect_analysis_corrected.csv")
write.csv(result_table, csv_file, row.names = FALSE, fileEncoding = "UTF-8")

# Optional publication-style Word table.
word_ok <- requireNamespace("officer", quietly = TRUE) &&
  requireNamespace("flextable", quietly = TRUE)

if (word_ok) {
  suppressPackageStartupMessages({
    library(officer)
    library(flextable)
  })

  ft <- flextable(result_table) |>
    set_header_labels(
      Parameter = "Parameter",
      MeCT_OR = "OR (95% CI)", MeCT_P = "P value",
      SFT_OR = "OR (95% CI)", SFT_P = "P value"
    ) |>
    add_header_row(
      values = c(
        "Parameter",
        "Mean CT attenuation\n(per 20 HU)",
        "Subcutaneous fat thickness\n(per 5 mm)"
      ),
      colwidths = c(1, 2, 2)
    ) |>
    merge_v(j = "Parameter", part = "header") |>
    theme_booktabs() |>
    bg(part = "header", bg = "#D9D9D9") |>
    bg(i = 3:5, bg = "#DDEBF7", part = "body") |>
    bold(i = 3:5, bold = TRUE, part = "body") |>
    bold(part = "header") |>
    font(fontname = "Times New Roman", part = "all") |>
    fontsize(size = 10, part = "all") |>
    align(j = "Parameter", align = "left", part = "body") |>
    align(j = 2:5, align = "center", part = "all") |>
    valign(valign = "center", part = "all") |>
    width(j = 1, width = 3.25) |>
    width(j = 2:5, width = 1.15)

  note <- paste0(
    "Note: Both breakpoints were estimated by a two-stage maximum-likelihood ",
    "grid search restricted to the 10th-90th percentile range. Scale-specific ",
    "coarse and refined step sizes are reported in the Methods. Davies tests ",
    "evaluated the global null hypothesis of no change in slope. Segment-specific ",
    "ORs, Wald 95% CIs, and Wald P values are conditional on the estimated ",
    "breakpoint. The MeCT model was adjusted for stone volume, ureteral wall ",
    "thickness, hydronephrosis, SFT, and stone location; the SFT model was ",
    "adjusted for stone volume, ureteral wall thickness, hydronephrosis, MeCT, ",
    "and stone location."
  )

  doc <- read_docx() |>
    body_add_par(
      "Table. Threshold effect analysis of mean CT attenuation and subcutaneous fat thickness on ESWL success",
      style = "heading 1"
    ) |>
    body_add_flextable(ft) |>
    body_add_par(
      "Abbreviations: CI, confidence interval; ESWL, extracorporeal shock-wave lithotripsy; HU, Hounsfield unit; OR, odds ratio."
    ) |>
    body_add_par(note) |>
    body_end_section_landscape()

  print(
    doc,
    target = file.path(output_dir, "Table2_threshold_effect_analysis_corrected.docx")
  )
}

print(result_table)

