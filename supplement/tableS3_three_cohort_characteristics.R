# Table S3: six predictors and two outcomes in three cohorts.
# Select the composite-success RData first, then the simple-success RData.
# The two workspaces are loaded into separate environments.

packages <- c("officer", "flextable")
missing_packages <- packages[!vapply(packages, requireNamespace,
                                   logical(1), quietly = TRUE)]
if (length(missing_packages)) install.packages(missing_packages)
library(officer)
library(flextable)

# 1. Input and output -----------------------------------------------------
composite_file <- "data/复合碎石成功的三个队列.RData"
simple_file <- "data/单纯碎石成功的三队列.RDATA"
output_dir <- "output/TableS3"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

composite_env <- new.env()
simple_env <- new.env()
load(composite_file, envir = composite_env)
load(simple_file, envir = simple_env)

get_data <- function(env, name) {
  if (!exists(name, envir = env, inherits = FALSE)) {
    stop("Object missing: ", name, ". Check selected RData file.")
  }
  dat <- get(name, envir = env, inherits = FALSE)
  if (!is.data.frame(dat)) stop("Not a data frame: ", name)
  dat
}

cohorts <- list(
  Training = get_data(composite_env, "training_data"),
  Internal_test = get_data(composite_env, "test_data"),
  External_validation = get_data(composite_env, "external_data")
)
simple_cohorts <- list(
  Training = get_data(simple_env, "training_data_sfr"),
  Internal_test = get_data(simple_env, "test_data_sfr"),
  External_validation = get_data(simple_env, "external_data_sfr")
)

continuous <- c("Mean CT attenuation", "Stone volume",
                "Ureteral wall thickness", "Subcutaneous fat thickness",
                "hydronephrosis")
predictors <- c(continuous, "Stone location")
location_levels <- c("Distal ureter", "Mid ureter", "Proximal ureter")
expected_n <- c(621L, 208L, 147L)

as_numeric_checked <- function(x, variable) {
  out <- suppressWarnings(as.numeric(as.character(x)))
  if (any(!is.na(x) & !is.finite(out))) {
    stop("Invalid numeric values: ", variable)
  }
  out
}

as_success <- function(x) {
  x <- trimws(as.character(x))
  if (anyNA(x) || !all(x %in% c("No", "Yes"))) {
    stop("Outcome must be complete and encoded as No/Yes.")
  }
  as.integer(x == "Yes")
}

# Verify cohort sizes, columns and predictor alignment across workspaces.
for (i in seq_along(cohorts)) {
  a <- cohorts[[i]]
  b <- simple_cohorts[[i]]
  required <- c(predictors, "success")
  if (!all(required %in% names(a)) || !all(required %in% names(b))) {
    stop("Missing required columns in cohort: ", names(cohorts)[i])
  }
  if (nrow(a) != expected_n[i] || nrow(b) != expected_n[i]) {
    stop("Unexpected cohort size: ", names(cohorts)[i])
  }
  for (v in continuous) {
    a[[v]] <- as_numeric_checked(a[[v]], v)
    b[[v]] <- as_numeric_checked(b[[v]], v)
    if (!isTRUE(all.equal(a[[v]], b[[v]], tolerance = 1e-10,
                         check.attributes = FALSE))) {
      stop("Predictor values or row order differ between workspaces: ", v)
    }
  }
  a[["Stone location"]] <- trimws(as.character(a[["Stone location"]]))
  b[["Stone location"]] <- trimws(as.character(b[["Stone location"]]))
  if (!identical(a[["Stone location"]], b[["Stone location"]])) {
    stop("Stone location or row order differs between workspaces.")
  }
  if (anyNA(a[predictors])) stop("Missing predictor values in: ", names(cohorts)[i])
  if (!all(a[["Stone location"]] %in% location_levels)) {
    stop("Unexpected stone-location labels.")
  }
  a$composite_success <- as_success(a$success)
  a$simple_success <- as_success(b$success)
  cohorts[[i]] <- a
}

# 2. Descriptive statistics and SMD ---------------------------------------
median_iqr <- function(x) {
  q <- quantile(x, c(0.50, 0.25, 0.75), names = FALSE, type = 7)
  sprintf("%.2f (%.2f–%.2f)", q[1], q[2], q[3])
}
count_pct <- function(x, value) {
  n <- sum(x == value)
  sprintf("%d (%.1f)", n, 100 * n / length(x))
}
smd_continuous <- function(x, y) {
  difference <- abs(mean(x) - mean(y))
  denominator <- sqrt((var(x) + var(y)) / 2)
  if (denominator == 0) return(if (difference == 0) 0 else Inf)
  difference / denominator
}
smd_binary <- function(x, y) {
  p <- mean(x)
  q <- mean(y)
  denominator <- sqrt((p * (1 - p) + q * (1 - q)) / 2)
  if (denominator == 0) return(if (p == q) 0 else Inf)
  abs(p - q) / denominator
}

# Overall multicategory SMD based on average multinomial covariance.
# Drop one category to avoid the redundant sum-to-one dimension.
smd_location <- function(x, y) {
  p <- as.numeric(table(factor(x, levels = location_levels))) / length(x)
  q <- as.numeric(table(factor(y, levels = location_levels))) / length(y)
  keep <- (p + q) > 0
  p <- p[keep]
  q <- q[keep]
  if (length(p) < 2L) return(0)
  p <- p[-length(p)]
  q <- q[-length(q)]
  covariance <- (diag(p, nrow = length(p)) - tcrossprod(p) +
                 diag(q, nrow = length(q)) - tcrossprod(q)) / 2
  e <- eigen(covariance, symmetric = TRUE)
  difference <- as.vector(crossprod(e$vectors, p - q))
  positive <- e$values > 1e-12
  if (any(abs(difference[!positive]) > 1e-10)) return(Inf)
  sqrt(sum(difference[positive]^2 / e$values[positive]))
}

labels <- c(
  "Mean CT attenuation (MeCT), HU",
  "Stone volume (SV), mm³",
  "Ureteral wall thickness (UWT), mm",
  "Subcutaneous fat thickness (SFT), mm",
  "Hydronephrosis measurement (MH), mm"
)
rows <- list()
add_row <- function(label, values = rep("", 3), smd = c(NA_real_, NA_real_),
                    section = FALSE) {
  rows[[length(rows) + 1L]] <<- data.frame(
    Characteristic = label,
    Training = values[1], Internal_test = values[2],
    External_validation = values[3],
    SMD_test_value = smd[1], SMD_external_value = smd[2],
    Section = section, stringsAsFactors = FALSE
  )
}

add_row("Model predictors", section = TRUE)
for (i in seq_along(continuous)) {
  v <- continuous[i]
  values <- vapply(cohorts, function(d) median_iqr(d[[v]]), character(1))
  smd <- c(smd_continuous(cohorts[[1]][[v]], cohorts[[2]][[v]]),
           smd_continuous(cohorts[[1]][[v]], cohorts[[3]][[v]]))
  add_row(labels[i], values, smd)
}

v <- "Stone location"
add_row("Stone location (SL), n (%)", smd = c(
  smd_location(cohorts[[1]][[v]], cohorts[[2]][[v]]),
  smd_location(cohorts[[1]][[v]], cohorts[[3]][[v]])
))
for (level in location_levels) {
  values <- vapply(cohorts, function(d) count_pct(d[[v]], level), character(1))
  add_row(paste0("  ", level), values)
}

add_row("Outcomes", section = TRUE)
outcomes <- c("composite_success", "simple_success")
outcome_labels <- c("Composite ESWL success, n (%)",
                    "Stone-control success, n (%)")
for (i in seq_along(outcomes)) {
  v <- outcomes[i]
  values <- vapply(cohorts, function(d) count_pct(d[[v]], 1L), character(1))
  smd <- c(smd_binary(cohorts[[1]][[v]], cohorts[[2]][[v]]),
           smd_binary(cohorts[[1]][[v]], cohorts[[3]][[v]]))
  add_row(outcome_labels[i], values, smd)
}

audit <- do.call(rbind, rows)
audit$SMD_test <- ifelse(is.na(audit$SMD_test_value), "",
                         sprintf("%.3f", audit$SMD_test_value))
audit$SMD_external <- ifelse(is.na(audit$SMD_external_value), "",
                             sprintf("%.3f", audit$SMD_external_value))
table_s3 <- audit[c("Characteristic", "Training", "Internal_test",
                    "External_validation", "SMD_test", "SMD_external")]

# 3. Publication-style Word table ----------------------------------------
ft <- flextable(table_s3)
ft <- set_header_labels(ft,
  Characteristic = "Characteristic",
  Training = "Training\n(n=621)",
  Internal_test = "Internal test\n(n=208)",
  External_validation = "External validation\n(n=147)",
  SMD_test = "SMD\nTraining vs test",
  SMD_external = "SMD\nTraining vs external")
ft <- theme_booktabs(ft)
ft <- font(ft, fontname = "Times New Roman", part = "all")
ft <- fontsize(ft, size = 10, part = "all")
ft <- bold(ft, part = "header")
ft <- align(ft, j = 1, align = "left", part = "all")
ft <- align(ft, j = 2:6, align = "center", part = "all")
ft <- valign(ft, valign = "center", part = "all")
ft <- padding(ft, padding.top = 5, padding.bottom = 5, part = "all")
ft <- width(ft, j = 1:6, width = c(2.95, 1.55, 1.55, 1.55, 1.10, 1.10))
ft <- set_table_properties(ft, layout = "fixed", opts_word = list(split = FALSE))

section_rows <- which(audit$Section)
for (i in section_rows) ft <- merge_at(ft, i = i, j = 1:6, part = "body")
ft <- bold(ft, i = section_rows, part = "body")
ft <- bg(ft, i = section_rows, bg = "#F2F2F2", part = "body")
ft <- align(ft, i = section_rows, align = "left", part = "body")
ft <- bold(ft, i = which(!is.na(audit$SMD_test_value) & audit$SMD_test_value >= 0.10),
           j = "SMD_test", part = "body")
ft <- bold(ft, i = which(!is.na(audit$SMD_external_value) & audit$SMD_external_value >= 0.10),
           j = "SMD_external", part = "body")

title <- "Table S3. Baseline characteristics and outcomes of the training, internal test, and external validation cohorts"
note <- paste(
  "Continuous variables are presented as median (IQR); categorical variables as n (%).",
  "SMDs are absolute standardized mean differences. Continuous-variable SMDs were calculated using means and variances on the original scale; one overall SMD is shown for stone location.",
  "SMDs ≥0.10 are shown in bold. The six model predictors were used in the prediction model; outcomes were not used as predictors.",
  "Composite ESWL success and stone-control success were defined as in the Methods."
)
abbreviations <- paste(
  "Abbreviations: CT, computed tomography; ESWL, extracorporeal shock wave lithotripsy;",
  "HU, Hounsfield unit; IQR, interquartile range; MeCT, mean CT attenuation;",
  "MH, hydronephrosis measurement; SFT, subcutaneous fat thickness; SL, stone location;",
  "SMD, standardized mean difference; SV, stone volume; UWT, ureteral wall thickness."
)

doc <- read_docx()
doc <- body_add_fpar(doc, fpar(ftext(title,
  fp_text(font.family = "Times New Roman", font.size = 11, bold = TRUE))))
doc <- body_add_flextable(doc, ft)
for (text in c(note, abbreviations)) {
  doc <- body_add_fpar(doc, fpar(ftext(text,
    fp_text(font.family = "Times New Roman", font.size = 9))))
}
doc <- body_set_default_section(doc, prop_section(
  page_size = page_size(orient = "landscape", width = 8.5, height = 11),
  page_margins = page_mar(top = 0.55, bottom = 0.55, left = 0.55, right = 0.55)))

word_path <- file.path(output_dir, "TableS3_three_cohorts_from_RData.docx")
print(doc, target = word_path)
write.csv(table_s3, file.path(output_dir, "TableS3_three_cohorts_from_RData.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")
write.csv(audit, file.path(output_dir, "TableS3_SMD_audit.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")
print(table_s3, row.names = FALSE)
message("Saved: ", word_path)
