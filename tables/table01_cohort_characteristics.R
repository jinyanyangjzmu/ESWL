# Table 1: training versus internal test cohort
# Run in R 4.3 or later. The fixed split is read from output/feature_selection/.

packages <- c("officer", "flextable")
missing_packages <- packages[!vapply(packages, requireNamespace,
                                   logical(1), quietly = TRUE)]
if (length(missing_packages)) install.packages(missing_packages)
library(officer)
library(flextable)

# 1. Paths ---------------------------------------------------------------
train_file <- "output/feature_selection/01_traindata.rds"
test_file <- "output/feature_selection/01_testdata_SEALED.rds"
output_dir <- "output/Table1"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

train <- readRDS(train_file)
test  <- readRDS(test_file)
stopifnot(is.data.frame(train), is.data.frame(test))
if (nrow(train) != 621L || nrow(test) != 208L) {
  stop("Expected training n=621 and internal test n=208; check selected files.")
}

# 2. Variable order, type and publication labels -------------------------
# C = continuous; B = binary (show Yes only); F = categorical (all levels).
spec <- data.frame(
  variable = c(
    "Age", "Sex", "Drink", "Smoke", "BMI", "Body temperature",
    "pain duration", "VAS score", "Diabetes mellitus",
    "Preoperative intervention treatment", "Stone laterality",
    "Prior SWL before presentation", "Oral medical expulsive therapy, MET",
    "Nausea/vomiting",
    "Stone length", "Stone width", "Stone depth", "Stone volume",
    "Stone area", "Skin to stone distance", "Mean CT attenuation",
    "Maximum CT attenuation", "Ureteral wall thickness",
    "Degree of hydronephrosis", "Subcutaneous fat thickness",
    "Stone heterogeneity index", "Coefficient of variation of stone density",
    "Number of stones", "Stone location",
    "Treatment duration", "Shockwave energy", "Number of shocks"
  ),
  label = c(
    "Age, years", "Sex, n (%)", "Alcohol use, n (%)", "Smoking, n (%)",
    "BMI, kg/m², n (%)", "Body temperature, °C", "Pain duration, h, n (%)",
    "VAS score, n (%)", "Diabetes mellitus, n (%)",
    "Preoperative intervention, n (%)", "Stone laterality, n (%)",
    "Prior SWL before presentation, n (%)", "Medical expulsive therapy, n (%)",
    "Nausea/vomiting, n (%)",
    "Stone length, mm", "Stone width, mm", "Stone depth, mm",
    "Stone volume, mm³", "Stone surface area, mm²", "Skin-to-stone distance, mm",
    "Mean CT attenuation, HU", "Maximum CT attenuation, HU",
    "Ureteral wall thickness, mm", "Hydronephrosis measurement, mm",
    "Subcutaneous fat thickness, mm", "Stone heterogeneity index, HU",
    "Coefficient of variation of stone density, %", "Number of stones, n (%)",
    "Stone location, n (%)", "Treatment duration, min, n (%)",
    "Shock-wave voltage setting, kV", "Number of shocks"
  ),
  type = c(
    "C", "F", "B", "B", "F", "C", "F", "F", "B", "B", "F", "B", "B", "B",
    rep("C", 13), "F", "F", "F", "C", "C"
  ),
  section = c(rep(1L, 14), rep(2L, 15), rep(3L, 3)),
  stringsAsFactors = FALSE
)

absent <- union(setdiff(spec$variable, names(train)),
                setdiff(spec$variable, names(test)))
if (length(absent)) stop("Variables missing: ", paste(absent, collapse = ", "))

# Standardize labels without changing the underlying categories.
clean_category <- function(x, variable) {
  x <- trimws(as.character(x))
  x <- gsub("[–—−]", "-", x)
  x[x == ""] <- NA_character_
  if (variable == "BMI") {
    x <- sub("^Normal \\((.*)\\)$", "\\1", x)
    x <- sub("^Overweight \\((.*)\\)$", "\\1", x)
    x <- sub("^Obesity \\((.*)\\)$", "\\1", x)
    x <- sub("^Underweight \\((.*)\\)$", "\\1", x)
  }
  if (variable == "Treatment duration") x[x == "60以上"] <- ">60"
  x
}

preferred_levels <- list(
  Sex = c("Male", "Female"),
  BMI = c("18.5-24.9", "25.0-29.9", ">30.0", "<18.5"),
  "pain duration" = c("0-24", "24-48", ">48"),
  "VAS score" = c("7-10", "4-6", "1-3", "0"),
  "Stone laterality" = c("Left", "Right"),
  "Number of stones" = c("single", "multiple"),
  "Stone location" = c("Distal ureter", "Proximal ureter", "Mid ureter"),
  "Treatment duration" = c("40-60", "20-40", ">60", "0-20")
)

for (i in seq_len(nrow(spec))) {
  v <- spec$variable[i]
  if (spec$type[i] == "C") {
    for (nm in c("train", "test")) {
      dat <- get(nm)
      old <- dat[[v]]
      new <- suppressWarnings(as.numeric(as.character(old)))
      if (any(!is.na(old) & !is.finite(new))) {
        stop("Invalid continuous values in ", nm, ": ", v)
      }
      dat[[v]] <- new
      assign(nm, dat)
    }
  } else {
    train[[v]] <- clean_category(train[[v]], v)
    test[[v]]  <- clean_category(test[[v]], v)
    if (spec$type[i] == "B") {
      observed <- na.omit(c(train[[v]], test[[v]]))
      if (!all(observed %in% c("No", "Yes"))) {
        stop("Expected No/Yes coding for: ", v)
      }
    }
  }
}

# 3. Descriptive statistics and absolute SMDs -----------------------------
median_iqr <- function(x) {
  x <- x[!is.na(x)]
  if (!length(x)) return("NA")
  q <- quantile(x, c(0.50, 0.25, 0.75), names = FALSE, type = 7)
  sprintf("%.2f (%.2f-%.2f)", q[1], q[2], q[3])
}

count_pct <- function(x, level) {
  denominator <- sum(!is.na(x))
  if (!denominator) return("NA")
  n <- sum(x == level, na.rm = TRUE)
  sprintf("%d (%.1f%%)", n, 100 * n / denominator)
}

smd_continuous <- function(x, y) {
  x <- x[!is.na(x)]
  y <- y[!is.na(y)]
  if (length(x) < 2L || length(y) < 2L) return(NA_real_)
  difference <- abs(mean(x) - mean(y))
  denominator <- sqrt((var(x) + var(y)) / 2)
  if (denominator == 0) return(if (difference == 0) 0 else Inf)
  difference / denominator
}

# Multicategory SMD: Mahalanobis distance using the average of the two
# multinomial covariance matrices. Binary variables use the same formula.
smd_categorical <- function(x, y, levels) {
  x <- x[!is.na(x)]
  y <- y[!is.na(y)]
  if (!length(x) || !length(y)) return(NA_real_)
  p <- as.numeric(table(factor(x, levels = levels))) / length(x)
  q <- as.numeric(table(factor(y, levels = levels))) / length(y)
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

section_labels <- c(
  "Demographic and pretreatment clinical characteristics",
  "Stone and CT characteristics",
  "ESWL treatment/procedure characteristics"
)
rows <- list()
add_row <- function(label, a = "", b = "", smd = NA_real_, section = FALSE) {
  rows[[length(rows) + 1L]] <<- data.frame(
    Characteristic = label, Training = a, Internal_test = b,
    SMD_value = smd, Section = section, stringsAsFactors = FALSE
  )
}

for (s in seq_along(section_labels)) {
  add_row(section_labels[s], section = TRUE)
  for (i in which(spec$section == s)) {
    v <- spec$variable[i]
    x <- train[[v]]
    y <- test[[v]]
    if (spec$type[i] == "C") {
      add_row(spec$label[i], median_iqr(x), median_iqr(y),
              smd_continuous(x, y))
    } else {
      observed <- unique(na.omit(c(x, y)))
      levels <- if (spec$type[i] == "B") c("No", "Yes") else {
        preferred <- preferred_levels[[v]]
        c(preferred, setdiff(observed, preferred))
      }
      smd <- smd_categorical(x, y, levels)
      if (spec$type[i] == "B") {
        add_row(spec$label[i], count_pct(x, "Yes"), count_pct(y, "Yes"), smd)
      } else {
        add_row(spec$label[i], smd = smd)
        for (level in levels) {
          add_row(paste0("  ", level), count_pct(x, level), count_pct(y, level))
        }
      }
    }
    if (anyNA(x) || anyNA(y)) {
      add_row("  Missing, n (%)",
              sprintf("%d (%.1f%%)", sum(is.na(x)), 100 * mean(is.na(x))),
              sprintf("%d (%.1f%%)", sum(is.na(y)), 100 * mean(is.na(y))))
    }
  }
}

result <- do.call(rbind, rows)
result$SMD <- ifelse(is.na(result$SMD_value), "",
                     sprintf("%.3f", result$SMD_value))
table1 <- result[c("Characteristic", "Training", "Internal_test", "SMD")]

# 4. Word table -----------------------------------------------------------
ft <- flextable(table1)
ft <- set_header_labels(ft,
  Characteristic = "Characteristic",
  Training = sprintf("Training (n=%d)", nrow(train)),
  Internal_test = sprintf("Internal test (n=%d)", nrow(test)),
  SMD = "SMD")
ft <- theme_booktabs(ft)
ft <- font(ft, fontname = "Times New Roman", part = "all")
ft <- fontsize(ft, size = 10, part = "all")
ft <- bold(ft, part = "header")
ft <- align(ft, j = 1, align = "left", part = "all")
ft <- align(ft, j = 2:4, align = "center", part = "all")
ft <- valign(ft, valign = "center", part = "all")
ft <- padding(ft, padding.top = 3, padding.bottom = 3, part = "all")
ft <- width(ft, j = 1:4, width = c(3.2, 1.55, 1.55, 0.55))
ft <- set_table_properties(ft, layout = "fixed", opts_word = list(
  split = FALSE, repeat_headers = TRUE))

section_rows <- which(result$Section)
for (i in section_rows) ft <- merge_at(ft, i = i, j = 1:4, part = "body")
ft <- bold(ft, i = section_rows, part = "body")
ft <- bg(ft, i = section_rows, bg = "#F2F2F2", part = "body")
ft <- align(ft, i = section_rows, align = "left", part = "body")
ft <- bold(ft, i = which(!is.na(result$SMD_value) & result$SMD_value >= 0.10),
           j = "SMD", part = "body")

legend <- paste(
  "Values are median (IQR) for continuous variables and n (%) for categorical variables.",
  "For binary variables, the values represent the presence of the characteristic.",
  "Percentages use nonmissing observations; missing values, if any, are reported separately using cohort size as the denominator.",
  "SMDs are absolute standardized mean differences; continuous-variable SMDs were calculated on the original scale, and one overall SMD is shown for each multicategory variable.",
  "SMDs ≥0.10 are shown in bold.",
  "Treatment/procedure characteristics are reported descriptively and were not used as pretreatment model predictors.",
  "The measurement unit for shockwave energy must be confirmed from the ESWL device record.",
  "Abbreviations: BMI, body mass index; CT, computed tomography; ESWL, extracorporeal shock wave lithotripsy; HU, Hounsfield unit; IQR, interquartile range; SMD, standardized mean difference; SWL, shock wave lithotripsy; VAS, visual analogue scale."
)

doc <- read_docx()
doc <- body_add_fpar(doc, fpar(ftext(
  "Table 1. Baseline and treatment characteristics of the training and internal test cohorts",
  fp_text(font.family = "Times New Roman", font.size = 11, bold = TRUE))))
doc <- body_add_flextable(doc, ft)
doc <- body_add_fpar(doc, fpar(ftext(legend,
  fp_text(font.family = "Times New Roman", font.size = 9))))
doc <- body_set_default_section(doc, prop_section(
  page_size = page_size(width = 8.5, height = 11),
  page_margins = page_mar(top = 0.65, bottom = 0.65, left = 0.65, right = 0.65)))

word_path <- file.path(output_dir, "Table1_baseline_from_RDS.docx")
print(doc, target = word_path)
write.csv(table1, file.path(output_dir, "Table1_baseline_from_RDS.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")
write.csv(result, file.path(output_dir, "Table1_SMD_audit.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")
print(table1, row.names = FALSE)
message("Saved: ", word_path)
