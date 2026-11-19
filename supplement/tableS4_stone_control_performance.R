# ======================================================================
# 单纯碎石成功：沿用既有 XGBoost 及固定 OOF 阈值，仅输出一张性能表
# 这是原模型对单纯碎石成功结局的评价，不是重新训练的新结局模型。
# 输出：TableS4_XGBoost_Stone_Control.csv；不输出图片或其他结果文件。
# 建议在保存模型时的原 R / xgboost / parsnip 环境运行。
# ======================================================================

# 1. 文件路径：修改；路径不存在时依次弹窗选择相应文件 -------------------
input_dir <- "."
model_file <- file.path(input_dir, "output/model_benchmark_9_CV_FORMAL/13_final_models/00_ESWL_all_9_best_CV_refit_models.rds")
data_file <- file.path(input_dir, "data/单纯碎石成功的三队列.RDATA")
threshold_file <- file.path(input_dir, "output/model_benchmark_9_CV_FORMAL/04_development_OOF_Youden_thresholds.csv")
if (!file.exists(model_file)) model_file <- file.choose()     # 模型 RDS
if (!file.exists(data_file)) data_file <- file.choose()       # 单纯成功 RDATA
if (!file.exists(threshold_file)) threshold_file <- file.choose() # 固定阈值 CSV
output_dir <- "output/TableS4"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_file <- file.path(output_dir, "TableS4_XGBoost_Stone_Control.csv")
B <- 2000L
set.seed(20260905)

pkgs <- c("workflows", "parsnip", "recipes", "hardhat", "xgboost", "pROC", "rlang")
missing_pkgs <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs)) stop("请在原建模环境安装缺少的包：", paste(missing_pkgs, collapse = ", "))
suppressPackageStartupMessages({
  library(workflows); library(parsnip); library(recipes); library(xgboost)
})

# 2. 读取已固定的阈值；不存在任何重新优化阈值的逻辑 ---------------------
th <- read.csv(threshold_file, stringsAsFactors = FALSE, fileEncoding = "UTF-8-BOM")
stopifnot(all(c("Model", "OOF_Youden_threshold") %in% names(th)))
ix <- which(trimws(th$Model) == "XGBoost")
if (length(ix) != 1L) stop("阈值文件中应当恰好有一行 XGBoost。")
threshold <- as.numeric(th$OOF_Youden_threshold[ix])
if (!is.finite(threshold) || threshold <= 0 || threshold >= 1)
  stop("固定阈值无效；请检查阈值文件。")
# 本次上传文件的固定阈值：0.7307063937187195，按完整精度使用。

# 3. 提取现有模型及三个单纯成功队列 ------------------------------------
obj <- readRDS(model_file)
wf <- if (inherits(obj, "workflow")) obj else obj$models[["XGBoost"]]
if (!inherits(wf, "workflow")) stop("未找到已拟合的 XGBoost workflow。")
fit <- workflows::extract_fit_parsnip(wf)
stopifnot(identical(as.character(fit$lvl), c("1", "0")))
event_level <- fit$spec$eng_args$event_level
if (!is.null(event_level) && !identical(rlang::eval_tidy(event_level), "first"))
  stop("模型阳性类别与本脚本不一致。")
# 已核对本次模型的最佳轮数为 68；保留的总轮数为 93。
engine <- workflows::extract_fit_engine(wf)
if (is.list(engine) && !is.null(engine$best_ntreelimit) && engine$best_ntreelimit != 68L)
  stop("模型最佳轮数与已核对的模型不同。")
best_rounds <- 68L

e <- new.env(parent = emptyenv())
load(data_file, envir = e)
object_names <- c("training_data_sfr", "test_data_sfr", "external_data_sfr")
if (!all(vapply(object_names, exists, logical(1), envir = e, inherits = FALSE)))
  stop("缺少 training_data_sfr、test_data_sfr 或 external_data_sfr。")
cohorts <- lapply(object_names, get, envir = e)
cohort_labels <- c("Training cohort", "Internal test cohort", "External validation cohort")
name_map <- c(
  "Stone volume" = "Stone_volume",
  "Mean CT attenuation" = "Mean_CT_attenuation",
  "Ureteral wall thickness" = "Ureteral_wall_thickness",
  "hydronephrosis" = "Hydronephrosis_measure",
  "Subcutaneous fat thickness" = "Subcutaneous_fat_thickness",
  "Stone location" = "Stone_location"
)

# 4. 指标与区间函数 ----------------------------------------------------
# AUC：DeLong；Accuracy/Sensitivity/Specificity/PPV/NPV：精确二项式 CI。
# 校准截距：glm(y ~ offset(logit(p)))，即斜率固定为 1。
# 校准斜率：glm(y ~ logit(p))，允许自由截距。
# 校准及 Brier：2000 次普通个体 bootstrap 百分位 CI。
# Bootstrap 只重抽样既有预测和结局，不重新训练 XGBoost 或选择阈值。
# 原开发队列表现不属于 OOF 性能，也未进行乐观偏倚校正。
exact_ci <- function(k, n) {
  if (n == 0) return(rep(NA_real_, 3))
  c(k/n, unname(binom.test(k, n)$conf.int))
}
safe_glm <- function(formula, data) {
  f <- suppressWarnings(tryCatch(
    glm(formula, data = data, family = binomial(), control = glm.control(maxit = 100)),
    error = function(e) NULL))
  if (is.null(f) || !isTRUE(f$converged) || isTRUE(f$boundary) || any(!is.finite(coef(f))))
    return(NULL)
  f
}
calibration <- function(y, p) {
  brier <- mean((y-p)^2)
  if (length(unique(y)) < 2L) return(c(NA_real_, NA_real_, brier))
  d <- data.frame(y = y, lp = qlogis(pmin(pmax(p, 1e-6), 1-1e-6)))
  f1 <- safe_glm(y ~ 1 + offset(lp), d)
  # 显式识别完全/准完全分离，避免将不可估计的斜率当作有效值。
  separated <- min(d$lp[y == 1]) >= max(d$lp[y == 0]) ||
               min(d$lp[y == 0]) >= max(d$lp[y == 1])
  f2 <- if (separated) NULL else safe_glm(y ~ lp, d)
  c(if (is.null(f1)) NA_real_ else unname(coef(f1)[1]),
    if (is.null(f2)) NA_real_ else unname(coef(f2)[2]), brier)
}
format_ci <- function(v) {
  if (!is.finite(v[1])) return("NE")
  if (any(!is.finite(v[2:3]))) return(sprintf("%.3f (CI not estimable)", v[1]))
  sprintf("%.3f (%.3f–%.3f)", v[1], v[2], v[3])
}

# 5. 逐队列计算，只生成表格中的十项指标 --------------------------------
columns <- lapply(seq_along(cohorts), function(i) {
  d <- as.data.frame(cohorts[[i]])
  names(d) <- trimws(names(d))
  stopifnot(all(c(names(name_map), "success") %in% names(d)))
  names(d)[match(names(name_map), names(d))] <- unname(name_map)
  truth <- trimws(as.character(d$success))
  if (anyNA(truth) || !all(truth %in% c("Yes", "No"))) stop("success 必须为 Yes/No。")
  y <- as.integer(truth == "Yes")
  if (length(unique(y)) != 2L) stop("每个队列均需包含成功和失败。")
  for (v in setdiff(unname(name_map), "Stone_location")) {
    d[[v]] <- suppressWarnings(as.numeric(as.character(d[[v]])))
    if (any(!is.finite(d[[v]]))) stop("预测变量存在缺失/非数值：", v)
  }
  loc <- trimws(as.character(d$Stone_location))
  lvls <- c("Distal ureter", "Mid ureter", "Proximal ureter")
  if (anyNA(loc) || !all(loc %in% lvls)) stop("Stone location 存在未知类别。")
  d$Stone_location <- factor(loc, levels = lvls)
  d$Case_ID <- seq_len(nrow(d)) # 仅满足保存 recipe 的 case_id 角色，不参与预测
  d$success <- NULL            # 新结局不传入预测过程

  # workflow 自动应用原有预处理；二分类 raw 概率对应首个类别 '1'。
  # R iterationrange 从 1 开始，上界不包含，因此 c(1,69) 使用最佳 68 轮。
  p <- as.numeric(predict(wf, new_data = d, type = "raw",
                         opts = list(iterationrange = c(1L, best_rounds + 1L))))
  n <- length(y)
  if (length(p) != n || any(!is.finite(p)) || any(p < 0 | p > 1)) stop("预测概率无效。")
  pred <- p >= threshold
  TN <- sum(y == 0 & !pred); FP <- sum(y == 0 & pred)
  FN <- sum(y == 1 & !pred); TP <- sum(y == 1 & pred)

  roc <- pROC::roc(y, p, levels = c(0,1), direction = "<", quiet = TRUE)
  aci <- as.numeric(pROC::ci.auc(roc, method = "delong"))
  point <- calibration(y, p)
  boot <- replicate(B, {
    j <- sample.int(n, n, replace = TRUE)
    calibration(y[j], p[j])
  })
  ci <- t(apply(boot, 1, function(x) {
    valid <- is.finite(x)
    if (mean(valid) < .95) return(c(NA_real_, NA_real_))
    unname(quantile(x[valid], c(.025, .975), type = 7))
  }))
  if (any(rowMeans(is.finite(boot)) < 1)) warning(
    cohort_labels[i], "：校准截距/斜率/Brier有效bootstrap次数为 ",
    paste(rowSums(is.finite(boot)), collapse = "/"), "，总次数为 ", B,
    "。有效比例低于95%的指标不报告CI。", call. = FALSE)

  stats <- list(
    c(as.numeric(pROC::auc(roc)), aci[1], aci[3]),
    exact_ci(TP+TN, n), exact_ci(TP, TP+FN), exact_ci(TN, TN+FP),
    exact_ci(TP, TP+FP), exact_ci(TN, TN+FN),
    c(point[1], ci[1,]), c(point[2], ci[2,]), c(point[3], ci[3,])
  )
  c(vapply(stats, format_ci, character(1)), paste(TN, FP, FN, TP, sep = " / "))
})

# 6. 唯一结果文件：Table S4 ----------------------------------------------
table3 <- data.frame(`Performance metric` = c(
  "AUC (95% CI)", "Accuracy (95% CI)", "Sensitivity (95% CI)",
  "Specificity (95% CI)", "PPV (95% CI)", "NPV (95% CI)",
  "Calibration intercept (95% CI)", "Calibration slope (95% CI)",
  "Brier score (95% CI)", "TN / FP / FN / TP"), check.names = FALSE)
for (i in seq_along(columns)) {
  table3[[sprintf("%s (n=%d)", cohort_labels[i], nrow(cohorts[[i]]))]] <- columns[[i]]
}
readr::write_excel_csv(table3, output_file, na = "")
print(table3, row.names = FALSE)
getwd()
