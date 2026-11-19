# ============================================================
# ESWL: fixed 75:25 split + conventional Boruta feature selection
# ============================================================

library(tidymodels)
library(readr)
library(dplyr)
library(Boruta)
library(ggplot2)

# ============================================================
# 1. File paths
# ============================================================

data_file <- "data/831ESWL.csv"
out <- "output/feature_selection"

dir.create(out, recursive = TRUE, showWarnings = FALSE)


# ============================================================
# 2. Read data
# ============================================================

ESWL <- readr::read_csv(
  data_file,
  show_col_types = FALSE
)

ESWL <- as.data.frame(
  ESWL,
  check.names = FALSE
)

names(ESWL) <- trimws(names(ESWL))


# ============================================================
# 3. Outcome
# ============================================================

if (!"success" %in% names(ESWL) &&
    "Stone success" %in% names(ESWL)) {
  ESWL$success <- ESWL[["Stone success"]]
}

if (!"success" %in% names(ESWL)) {
  stop("Data does not contain success or Stone success.")
}

ESWL$success <- factor(
  trimws(as.character(ESWL$success))
)

if (nlevels(ESWL$success) != 2L) {
  stop(
    "success must contain exactly two outcome levels. Current levels: ",
    paste(levels(ESWL$success), collapse = ", ")
  )
}

cat(
  "Outcome levels:",
  paste(levels(ESWL$success), collapse = ", "),
  "\n"
)


# ============================================================
# 4. Candidate variables
# ============================================================

candidate33 <- c(
  "Age",
  "Sex",
  "Drink",
  "Smoke",
  "Body temperature",
  "BMI",
  "pain duration",
  "VAS score",
  "Diabetes mellitus",
  "Number of stones",
  "Preoperative intervention treatment",
  "Stone laterality",
  "Treatment duration",
  "Prior SWL before presentation",
  "Oral medical expulsive therapy, MET",
  "Nausea/vomiting",
  "Stone location",
  "Stone length",
  "Stone width",
  "Stone depth",
  "Stone volume",
  "Stone area",
  "Skin to stone distance",
  "Shockwave energy",
  "Number of shocks",
  "Number of sessions",
  "Mean CT attenuation",
  "Maximum CT attenuation",
  "Ureteral wall thickness",
  "Degree of hydronephrosis",
  "Subcutaneous fat thickness",
  "Stone heterogeneity index",
  "Coefficient of variation of stone density"
)


# ============================================================
# 5. Exclude treatment/process variables
# ============================================================

exclude4 <- c(
  "Number of shocks",
  "Treatment duration",
  "Shockwave energy",
  "Number of sessions"
)

predictors <- setdiff(candidate33, exclude4)

cat(
  "Number of predictors entering Boruta:",
  length(predictors),
  "\n"
)


# ============================================================
# 6. Check variables
# ============================================================

missing_vars <- setdiff(
  c("success", predictors),
  names(ESWL)
)

if (length(missing_vars) > 0L) {
  stop(
    "The following variables are absent from the data:\n",
    paste(missing_vars, collapse = "\n")
  )
}


# ============================================================
# 7. Fixed 75:25 split
# ============================================================
# IMPORTANT:
# Feature selection is performed ONLY in the training set.
# The internal test set remains completely sealed.

set.seed(4321)

datasplit <- initial_split(
  ESWL,
  prop = 0.75,
  strata = success
)

traindata <- training(datasplit)
testdata  <- testing(datasplit)

cat("\nSample split:\n")
cat("Total    =", nrow(ESWL), "\n")
cat("Training =", nrow(traindata), "\n")
cat("Test     =", nrow(testdata), "\n")

cat("\nTraining outcome:\n")
print(table(traindata$success))
print(prop.table(table(traindata$success)))

cat("\nTest outcome:\n")
print(table(testdata$success))
print(prop.table(table(testdata$success)))


# Save fixed split
saveRDS(
  datasplit,
  file.path(out, "01_datasplit_seed4321.rds")
)

saveRDS(
  traindata,
  file.path(out, "01_traindata.rds")
)

saveRDS(
  testdata,
  file.path(out, "01_testdata_SEALED.rds")
)


# ============================================================
# 8. Variable types
# ============================================================

continuous <- c(
  "Age",
  "Body temperature",
  "Stone length",
  "Stone width",
  "Stone depth",
  "Stone volume",
  "Stone area",
  "Skin to stone distance",
  "Mean CT attenuation",
  "Maximum CT attenuation",
  "Ureteral wall thickness",
  "Degree of hydronephrosis",
  "Subcutaneous fat thickness",
  "Stone heterogeneity index",
  "Coefficient of variation of stone density"
)

continuous <- intersect(continuous, predictors)
categorical <- setdiff(predictors, continuous)


# Only use training data for Boruta
dat <- traindata[
  ,
  c("success", predictors),
  drop = FALSE
]


# Continuous variables
for (v in continuous) {
  dat[[v]] <- suppressWarnings(
    as.numeric(as.character(dat[[v]]))
  )
}


# Categorical variables
for (v in categorical) {
  dat[[v]] <- factor(
    trimws(as.character(dat[[v]]))
  )
}

dat$success <- droplevels(
  factor(dat$success)
)


# ============================================================
# 9. Complete-case analysis
# ============================================================

n_before <- nrow(dat)

dat <- dat[
  complete.cases(
    dat[, c("success", predictors), drop = FALSE]
  ),
  ,
  drop = FALSE
]

dat$success <- droplevels(dat$success)

cat(
  "\nTraining complete cases:",
  nrow(dat),
  "/",
  n_before,
  "\n"
)

if (nlevels(dat$success) != 2L) {
  stop(
    "The complete-case training data no longer contain two outcome levels."
  )
}

saveRDS(
  dat,
  file.path(out, "01_training_Boruta_complete_cases.rds")
)


# ============================================================
# 10. Conventional Boruta feature selection
# ============================================================

set.seed(688)

boruta_raw <- Boruta(
  x = dat[, predictors, drop = FALSE],
  y = dat$success,
  
  pValue = 0.01,
  mcAdj = TRUE,
  
  maxRuns = 500,
  
  doTrace = 2,
  holdHistory = TRUE,
  
  getImp = getImpRfZ,
  ntree = 500
)


# ============================================================
# 11. Resolve tentative variables
# ============================================================

cat("\nRaw Boruta decision table:\n")
print(
  table(boruta_raw$finalDecision)
)


# Only apply TentativeRoughFix when Tentative variables remain
if (any(boruta_raw$finalDecision == "Tentative")) {
  
  boruta_final <- TentativeRoughFix(
    boruta_raw
  )
  
} else {
  
  boruta_final <- boruta_raw
}


cat("\nFinal Boruta decision table:\n")
print(
  table(boruta_final$finalDecision)
)

cat(
  "Boruta importance runs:",
  nrow(boruta_raw$ImpHistory),
  "\n"
)


# ============================================================
# 12. Extract Boruta results
# ============================================================

st <- attStats(
  boruta_final
)

res <- data.frame(
  Variable = rownames(st),
  
  Mean_importance   = st$meanImp,
  Median_importance = st$medianImp,
  Min_importance    = st$minImp,
  Max_importance    = st$maxImp,
  
  Normalized_hits = st$normHits,
  
  Decision = as.character(st$decision),
  
  row.names = NULL,
  check.names = FALSE
) |>
  arrange(
    desc(Median_importance)
  )


confirmed <- res |>
  filter(
    Decision == "Confirmed"
  )

tentative <- res |>
  filter(
    Decision == "Tentative"
  )

rejected <- res |>
  filter(
    Decision == "Rejected"
  )


# ============================================================
# 13. Save Boruta results
# ============================================================

write.csv(
  res,
  file.path(
    out,
    "01_Boruta_results.csv"
  ),
  row.names = FALSE
)


write.csv(
  confirmed,
  file.path(
    out,
    "01_Boruta_confirmed.csv"
  ),
  row.names = FALSE
)


write.csv(
  tentative,
  file.path(
    out,
    "01_Boruta_tentative.csv"
  ),
  row.names = FALSE
)


write.csv(
  rejected,
  file.path(
    out,
    "01_Boruta_rejected.csv"
  ),
  row.names = FALSE
)


saveRDS(
  boruta_raw,
  file.path(
    out,
    "01_Boruta_raw_object.rds"
  )
)


saveRDS(
  boruta_final,
  file.path(
    out,
    "01_Boruta_final_object.rds"
  )
)


# ============================================================
# 14. Publication-style Boruta plot
# ============================================================

cols <- c(
  "Confirmed" = "#2A9D8F",
  "Tentative" = "#E9C46A",
  "Rejected"  = "#E76F51"
)


plot_data <- res |>
  mutate(
    Variable = factor(
      Variable,
      levels = rev(Variable)
    )
  )


p <- ggplot(
  plot_data,
  aes(
    x = Variable,
    y = Median_importance,
    fill = Decision
  )
) +
  
  geom_col(
    width = 0.72
  ) +
  
  coord_flip() +
  
  scale_fill_manual(
    values = cols,
    drop = FALSE
  ) +
  
  labs(
    x = NULL,
    y = "Median Boruta importance (Z-score)",
    fill = NULL,
    title = "Boruta feature selection"
  ) +
  
  theme_classic(
    base_size = 11,
    base_family = "sans"
  ) +
  
  theme(
    legend.position = "top",
    
    plot.title = element_text(
      face = "bold",
      hjust = 0.5,
      size = 13
    ),
    
    axis.text.y = element_text(
      size = 8.5
    ),
    
    axis.title.x = element_text(
      face = "bold"
    )
  )


# PDF
ggsave(
  filename = file.path(
    out,
    "Fig_Boruta.pdf"
  ),
  plot = p,
  device = grDevices::cairo_pdf,
  width = 8.5,
  height = 8.5,
  units = "in"
)


# TIFF 600 dpi
ggsave(
  filename = file.path(
    out,
    "Fig_Boruta.tiff"
  ),
  plot = p,
  width = 8.5,
  height = 8.5,
  units = "in",
  dpi = 600,
  compression = "lzw"
)


# ============================================================
# 15. Console output
# ============================================================

cat("\n============================================\n")
cat("Boruta Confirmed variables\n")
cat("============================================\n")

print(
  confirmed[
    ,
    c(
      "Variable",
      "Median_importance",
      "Normalized_hits"
    )
  ],
  row.names = FALSE
)

cat(
  "\nConfirmed variable count:",
  nrow(confirmed),
  "\n"
)

cat("\n============================================\n")
cat("Boruta Rejected variables\n")
cat("============================================\n")

print(
  rejected[
    ,
    c(
      "Variable",
      "Median_importance"
    )
  ],
  row.names = FALSE
)

cat(
  "\nResults directory:\n",
  out,
  "\n",
  sep = ""
)

cat(
  "\nIMPORTANT: The sealed internal test set was not used in Boruta feature selection.\n"
)