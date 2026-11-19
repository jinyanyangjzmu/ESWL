# Figure 4: Boruta importance and training-set predictor associations
# 图内不放标题、图例或缩写说明；这些内容请写入论文图注。
# 修改下面的目录，使其指向原始分析输出文件夹。

library(Boruta)
library(ggplot2)
library(patchwork)

out_dir <- "output/feature_selection"
train <- readRDS(file.path(out_dir, "01_traindata.rds"))
boruta <- readRDS(file.path(out_dir, "01_Boruta_raw_object.rds"))

if (nrow(train) != 621L) {
  stop("训练集应为621例；请检查01_traindata.rds。")
}

if (any(boruta$finalDecision == "Tentative")) {
  boruta <- TentativeRoughFix(boruta)
}

# A：使用已保存Boruta对象的中位重要性和最终判定
stats <- attStats(boruta)
importance <- data.frame(
  Variable = rownames(stats),
  Median = stats$medianImp,
  Decision = as.character(stats$decision),
  row.names = NULL
)
importance <- importance[order(importance$Median, decreasing = TRUE), ]
importance$Variable <- factor(
  importance$Variable,
  levels = rev(importance$Variable)
)

confirmed <- as.character(importance$Variable[importance$Decision == "Confirmed"])

expected <- c(
  "Mean CT attenuation", "Stone length", "Maximum CT attenuation",
  "Stone volume", "Stone heterogeneity index", "Stone depth",
  "Stone area", "Ureteral wall thickness", "Stone width",
  "Subcutaneous fat thickness", "Degree of hydronephrosis",
  "Stone location"
)

if (!setequal(confirmed, expected)) {
  stop("Boruta确认变量与预期的12项不一致；请核对上传的Boruta对象。")
}
if (!all(expected %in% names(train))) {
  stop("训练集缺少部分Boruta确认变量。")
}

p_a <- ggplot(importance, aes(x = Median, y = Variable, fill = Decision)) +
  geom_vline(xintercept = 0, colour = "#A9B2B9", linewidth = 0.35) +
  geom_col(width = 0.72) +
  scale_fill_manual(
    values = c(Confirmed = "#277C86", Rejected = "#B6BEC5"),
    guide = "none"
  ) +
  labs(x = "Median Boruta importance (Z score)", y = NULL) +
  theme_classic(base_size = 10, base_family = "Arial") +
  theme(
    axis.text.y = element_text(size = 8.5, colour = "#25343D"),
    axis.text.x = element_text(size = 9, colour = "#25343D"),
    axis.title.x = element_text(size = 10),
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    plot.margin = margin(12, 10, 12, 8)
  )

# B：仅使用621例训练集。连续变量之间用|Spearman rho|；
# Stone location与连续变量之间用相关比eta。
selected <- as.character(
  importance$Variable[importance$Decision == "Confirmed"]
)

numeric_vars <- setdiff(selected, "Stone location")
for (v in numeric_vars) {
  train[[v]] <- suppressWarnings(as.numeric(as.character(train[[v]])))
}
train[["Stone location"]] <- droplevels(
  factor(train[["Stone location"]])
)

association_ratio <- function(x, group) {
  keep <- is.finite(x) & !is.na(group)
  x <- x[keep]
  group <- droplevels(factor(group[keep]))
  if (length(x) < 3L || nlevels(group) < 2L || var(x) == 0) return(NA_real_)
  group_mean <- tapply(x, group, mean)
  sqrt(
    sum(as.numeric(table(group)) * (group_mean - mean(x))^2) /
      sum((x - mean(x))^2)
  )
}

strength <- matrix(
  1, length(selected), length(selected),
  dimnames = list(selected, selected)
)

for (i in seq_along(selected)) {
  for (j in seq_len(i - 1L)) {
    a <- selected[i]
    b <- selected[j]
    
    value <- if (a == "Stone location") {
      association_ratio(train[[b]], train[[a]])
    } else if (b == "Stone location") {
      association_ratio(train[[a]], train[[b]])
    } else {
      abs(cor(
        train[[a]], train[[b]],
        method = "spearman", use = "pairwise.complete.obs"
      ))
    }
    
    strength[i, j] <- value
    strength[j, i] <- value
  }
}

if (anyNA(strength)) {
  stop("相关性存在缺失值；请检查训练集中的确认变量。")
}

abbreviation <- c(
  "Mean CT attenuation" = "Mean CT",
  "Stone length" = "SL",
  "Maximum CT attenuation" = "Max CT",
  "Stone volume" = "SV",
  "Stone heterogeneity index" = "SHI",
  "Stone depth" = "SD",
  "Stone area" = "SA",
  "Ureteral wall thickness" = "UWT",
  "Stone width" = "SW",
  "Subcutaneous fat thickness" = "SFT",
  "Degree of hydronephrosis" = "Hyd.",
  "Stone location" = "Location"
)

ij <- which(lower.tri(strength, diag = TRUE), arr.ind = TRUE)
tiles <- data.frame(
  x = selected[ij[, 2]],
  y = selected[ij[, 1]],
  value = strength[ij]
)
tiles$x <- factor(tiles$x, levels = selected)
tiles$y <- factor(tiles$y, levels = rev(selected))
border_tiles <- tiles[
  as.character(tiles$x) != as.character(tiles$y) &
    tiles$value >= 0.80,
]

p_b <- ggplot(tiles, aes(x = x, y = y)) +
  geom_tile(aes(fill = value), colour = "white", linewidth = 0.45) +
  geom_tile(
    data = border_tiles,
    fill = NA, colour = "#24343D", linewidth = 0.75
  ) +
  geom_text(
    aes(label = sprintf("%.2f", value),
        colour = value >= 0.65),
    size = 2.65, family = "Arial", show.legend = FALSE
  ) +
  scale_colour_manual(
    values = c(`FALSE` = "#25343D", `TRUE` = "white"),
    guide = "none"
  ) +
  scale_fill_gradient(
    low = "#F1F5F5", high = "#206C80",
    limits = c(0, 1), guide = "none"
  ) +
  scale_x_discrete(labels = function(x) unname(abbreviation[x])) +
  scale_y_discrete(labels = function(x) unname(abbreviation[x])) +
  coord_equal(clip = "off") +
  labs(x = NULL, y = NULL) +
  theme_classic(base_size = 10, base_family = "Arial") +
  theme(
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    axis.text.x = element_text(
      angle = 55, hjust = 1, vjust = 1,
      size = 8.5, colour = "#25343D"
    ),
    axis.text.y = element_text(size = 9, colour = "#25343D"),
    plot.margin = margin(12, 15, 12, 4)
  )



tag_style <- theme(
  plot.tag = element_text(
    family = "Arial", face = "bold", size = 15,
    colour = "#25343D"
  )
)

figure4 <- p_a + p_b +
  plot_layout(widths = c(1.30, 1)) +
  plot_annotation(
    tag_levels = "A",
    theme = tag_style
  )

ggsave(
  file.path(out_dir, "Figure_4_training_only.pdf"),
  figure4,
  width = 17, height = 9.5, units = "in",
  device = grDevices::cairo_pdf
)

ggsave(
  file.path(out_dir, "Figure_4_training_only.tiff"),
  figure4,
  width = 17, height = 9.5, units = "in",
  dpi = 600, compression = "lzw"
)