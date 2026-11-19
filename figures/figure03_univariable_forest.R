# ESWL 单因素森林图：只生成一个 PDF，不导出表格、PNG 或 RData。
# 保留原分析：19变量；Yes为事件；整体LRT P<0.05入图；Wald 95%CI。

# 1. 设置：通常只需修改这两个路径
data_file <- "data/831ESWL.csv"
output_file <- "output/ESWL_univariable_significant_forest.pdf"

# 少量手动刻度 + 加宽绘图区，解决横轴数字挤在一起的问题。
axis_limits <- c(0.02, 20)
axis_ticks  <- c(0.02, 0.1, 0.5, 1, 5, 20)

pkgs <- c("readr", "forestploter")
missing_pkgs <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing_pkgs)) install.packages(missing_pkgs)

# 2. 读取数据；兼容修改前后的肾积水列名
dat <- as.data.frame(readr::read_csv(data_file, show_col_types = FALSE))
names(dat) <- trimws(names(dat))
old_name <- which(tolower(names(dat)) == "degree of hydronephrosis")
new_name <- which(tolower(names(dat)) == "hydronephrosis measurement")
if (length(old_name) && length(new_name)) stop("数据同时含新旧肾积水列名，请保留正确的一列。")
if (length(old_name)) names(dat)[old_name] <- "Hydronephrosis measurement"

variables <- c(
  "Maximum CT attenuation", "Mean CT attenuation", "Stone volume",
  "Stone heterogeneity index", "Stone depth", "Stone length", "Stone area",
  "Stone width", "Number of shocks", "Stone location",
  "Hydronephrosis measurement", "Treatment duration", "Ureteral wall thickness",
  "Number of stones", "Oral medical expulsive therapy, MET",
  "Subcutaneous fat thickness", "VAS score", "BMI",
  "Coefficient of variation of stone density"
)

# 忽略列名大小写；仅使用指定的19个变量。
required <- c("Stone success", variables)
columns <- vapply(required, function(v) {
  hit <- which(tolower(names(dat)) == tolower(v))
  if (length(hit) != 1) stop("找不到或重复字段：", v)
  hit
}, integer(1))
dat <- dat[, columns, drop = FALSE]
names(dat) <- required

# 连续变量的增量沿用原代码；NA表示分类变量。
steps <- c(20, 20, 100, 20, 1, 1, 10, 1, 100, NA, 1, NA, 1,
           NA, NA, 10, NA, NA, NA)
increments <- c(
  "Per 20 HU", "Per 20 HU", "Per 100 mm^3", "Per 20 HU",
  "Per 1 mm", "Per 1 mm", "Per 10 mm^2", "Per 1 mm",
  "Per 100 shocks", "", "Per 1 mm", "", "Per 1 mm", "", "",
  "Per 10 mm", "", "", ""
)

to_numeric <- function(x, label) {
  text <- trimws(as.character(x))
  text[text == ""] <- NA_character_
  value <- suppressWarnings(as.numeric(text))
  if (any(!is.na(text) & !is.finite(value))) {
    stop(label, " 含无法转换为有限数值的内容，请检查原始数据。")
  }
  value
}

# 保留原代码的CV尺度判定；若已知数据单位，请直接设置steps[19]。
cv <- to_numeric(dat[[variables[19]]], variables[19])
q95 <- if (all(is.na(cv))) NA_real_ else unname(quantile(abs(cv), 0.95, na.rm = TRUE))
fraction_scale <- is.finite(q95) && q95 <= 2
steps[19] <- if (fraction_scale) 0.1 else 10
increments[19] <- if (fraction_scale) "Per 0.1 increase" else "Per 10 percentage-point increase"

# 分类变量的参考顺序保持不变，包括原来的BMI低体重参考组。
clean <- function(x) gsub("[‐‑‒–—−]", "-", trimws(as.character(x)))
level_order <- list(
  BMI = c("Underweight (<18.5)", "Normal (18.5-24.9)",
          "Overweight (25.0-29.9)", "Obesity (>30.0)"),
  `VAS score` = c("0", "1-3", "4-6", "7-10"),
  `Treatment duration` = c("0-20", "20-40", "40-60", "60以上")
)
y <- trimws(as.character(dat[["Stone success"]]))
if (any(!is.na(y) & !y %in% c("No", "Yes"))) stop("Stone success必须为No/Yes。")
dat[["Stone success"]] <- factor(y, levels = c("No", "Yes"))

# 3. 分别拟合单因素模型，整理绘图数据（不导出结果表）
fit_one <- function(i) {
  v <- variables[i]
  continuous <- !is.na(steps[i])
  if (continuous) {
    x <- to_numeric(dat[[v]], v) / steps[i]
  } else {
    x <- clean(dat[[v]])
    if (v %in% names(level_order)) {
      unknown <- setdiff(x[!is.na(x)], level_order[[v]])
      if (length(unknown)) stop(v, " 存在未定义水平：", paste(unknown, collapse = ", "))
      x <- factor(x, levels = level_order[[v]])
    } else {
      x <- factor(x)
    }
  }
  z <- data.frame(y = dat[["Stone success"]], x = x)
  z <- z[complete.cases(z), , drop = FALSE]
  if (nrow(z) == 0 || length(unique(z$x)) < 2 || length(unique(z$y)) < 2) {
    warning("无法拟合：", v)
    return(NULL)
  }
  z <- droplevels(z)
  fit <- glm(y ~ x, data = z, family = binomial(),
             contrasts = if (!continuous) list(x = "contr.treatment") else NULL)
  if (!fit$converged) stop(v, " 的模型未收敛，请检查数据。")
  null <- glm(y ~ 1, data = z, family = binomial())
  overall_p <- anova(null, fit, test = "LRT")$`Pr(>Chi)`[2]
  if (!is.finite(overall_p) || overall_p >= 0.05) return(NULL)

  b <- summary(fit)$coefficients[-1, , drop = FALSE]
  if (any(!is.finite(b))) stop(v, " 的估计异常，请检查分离或稀疏类别。")
  result <- data.frame(
    Variable = v,
    Level = if (continuous) increments[i] else levels(z$x)[-1],
    OR = exp(b[, 1]), Lower = exp(b[, 1] - 1.96 * b[, 2]),
    Upper = exp(b[, 1] + 1.96 * b[, 2]), P = b[, 4],
    Overall_P = overall_p, Reference = FALSE, row.names = NULL
  )
  if (!continuous) {
    reference <- data.frame(Variable = v, Level = levels(z$x)[1],
                            OR = 1, Lower = 1, Upper = 1, P = NA_real_,
                            Overall_P = overall_p, Reference = TRUE)
    result <- rbind(reference, result)
  }
  result
}
results <- do.call(rbind, lapply(seq_along(variables), fit_one))
if (is.null(results)) stop("没有整体LRT P<0.05的变量，未生成森林图。")
results <- results[order(results$Overall_P), , drop = FALSE]

# 4. 文字和置信区间
fmt_p <- function(p) ifelse(is.na(p), "", ifelse(p < 0.001, "<0.001", sprintf("%.3f", p)))
fmt_or <- function(x) ifelse(x >= 100, sprintf("%.1f", x),
                           ifelse(x >= 10, sprintf("%.2f", x), sprintf("%.3f", x)))
wrap <- function(x, width) vapply(x, function(s) paste(strwrap(s, width), collapse = "\n"), character(1))
first <- !duplicated(results$Variable)
level_text <- gsub("60以上", "60 or more", results$Level, fixed = TRUE)
level_text[results$Reference] <- paste0(level_text[results$Reference], " (Reference)")
ci_text <- paste0(fmt_or(results$OR), " (", fmt_or(results$Lower), ", ", fmt_or(results$Upper), ")")
ci_text[results$Reference] <- "Reference"
plot_data <- data.frame(
  Variable = wrap(ifelse(first, results$Variable, ""), 30),
  `Level / increment` = wrap(level_text, 30),
  ` ` = strrep(" ", 52),  # 原来24个空格太窄；在建图前加宽
  `OR (95% CI)` = ci_text,
  `P value` = fmt_p(results$P),
  `Overall P` = ifelse(first, fmt_p(results$Overall_P), ""),
  check.names = FALSE
)
est <- results$OR; lower <- results$Lower; upper <- results$Upper
if (any(!is.finite(c(est, lower, upper))) || any(c(est, lower, upper) <= 0)) {
  stop("存在无穷大或零置信限，请先检查分离或稀疏类别，不能可靠绘图。")
}
est[results$Reference] <- lower[results$Reference] <- upper[results$Reference] <- NA_real_

# 5. 只保存一张PDF；根据整图实际宽高导出，避免左右文字被裁掉
save_forest <- function() {
  # 空设备只用于测量字体，不产生Rplots.pdf。
  grDevices::pdf(file = NULL, family = "Helvetica")
  on.exit(grDevices::dev.off(), add = TRUE)
  theme <- forestploter::forest_theme(
    base_size = 10, base_family = "Helvetica",
    ci_pch = 15, ci_col = "black", ci_lwd = 1.2,
    ci_Theight = grid::unit(1.5, "mm"),
    refline_gp = grid::gpar(col = "grey45", lty = 2),
    xaxis_gp = grid::gpar(fontsize = 9),
    xlab_gp = grid::gpar(fontsize = 10),
    footnote_gp = grid::gpar(fontsize = 8),
    core = list(bg_params = list(fill = c("#F1F5F4", "white")))
  )
  p <- forestploter::forest(
    plot_data, est = est, lower = lower, upper = upper,
    ci_column = 3, ref_line = 1, x_trans = "log",
    xlim = axis_limits, ticks_at = axis_ticks, ticks_digits = 2L,
    xlab = "Odds ratio (log scale)",
    theme = theme
  )
  p <- forestploter::edit_plot(p, part = "header", which = "text", gp = grid::gpar(fontface = "bold"))
  p <- forestploter::add_border(p, part = "header", where = "bottom", gp = grid::gpar(lwd = 1))
  size <- forestploter::get_wh(p, unit = "in")
  dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
  grDevices::pdf(output_file, width = size[1] + 0.5, height = size[2] + 0.4,
                 family = "Helvetica", useDingbats = FALSE)
  tryCatch({
    grid::grid.newpage()
    grid::grid.draw(p)
  }, finally = grDevices::dev.off())
  invisible(NULL)
}
save_forest()
