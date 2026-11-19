suppressPackageStartupMessages({
  library(rms)
  library(ggplot2)
  library(cowplot)
})

# ============================================================
# 1. 读取数据及设置输出目录
# ============================================================

raw <- read.csv(
  file = "data/831ESWL.csv",
  check.names = FALSE,
  stringsAsFactors = FALSE,
  fileEncoding = "UTF-8-BOM"
)

names(raw) <- trimws(names(raw))

output_dir <- "output/RCS_5variables"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# 2. 单位与单因素回归一致
#    Stone volume：100 mm³
#    Mean CT attenuation：20 HU
#    Ureteral wall thickness：1 mm
#    Hydronephrosis measurement：1 mm
#    Subcutaneous fat thickness：10 mm
# ============================================================

dt <- data.frame(
  success = match(
    trimws(raw[["Stone success"]]),
    c("No", "Yes")
  ) - 1L,
  
  SV = as.numeric(raw[["Stone volume"]]) / 100,
  MeCT = as.numeric(raw[["Mean CT attenuation"]]) / 20,
  UWT = as.numeric(raw[["Ureteral wall thickness"]]),
  HM = as.numeric(raw[["Degree of hydronephrosis"]]),
  SFT = as.numeric(raw[["Subcutaneous fat thickness"]]) / 10,
  
  SL = factor(
    trimws(raw[["Stone location"]]),
    levels = c("Distal ureter", "Mid ureter", "Proximal ureter")
  )
)

vars <- c("SV", "MeCT", "UWT", "HM", "SFT")

# 五个模型使用相同的完整病例
dt <- droplevels(dt[complete.cases(dt), ])

stopifnot(
  nrow(dt) > 0,
  length(unique(dt$success)) == 2L,
  nlevels(dt$SL) >= 2L,
  all(vapply(
    dt[vars],
    function(x) all(is.finite(x)) && length(unique(x)) >= 5L,
    logical(1)
  ))
)

axis_labels <- c(
  SV   = "Stone volume\n(100 mm\u00B3)",
  MeCT = "Mean CT attenuation\n(20 HU)",
  UWT  = "Ureteral wall thickness\n(mm)",
  HM   = "Hydronephrosis measurement\n(mm)",
  SFT  = "Subcutaneous fat thickness\n(10 mm)"
)

# ============================================================
# 3. 设置参照值
#    OR为相对于目标变量中位数的调整OR，
#    不是每增加一个单位的固定OR。
# ============================================================

old_datadist <- getOption("datadist")

dd_rcs <- datadist(dt)

for (v in vars) {
  dd_rcs$limits["Adjust to", v] <- median(dt[[v]])
}

options(datadist = "dd_rcs")

p_format <- function(p) {
  if (is.na(p)) return("= NA")
  if (p < 0.001) return("< 0.001")
  paste0("= ", sprintf("%.3f", p))
}

# ============================================================
# 4. 拟合模型并绘制五个RCS面板
#    每次目标变量采用4结点RCS，其余4个连续变量线性调整，
#    同时纳入分类变量Stone location。
# ============================================================

plots <- lapply(vars, function(v) {
  
  formula_rcs <- as.formula(paste(
    "success ~",
    paste(
      c(
        sprintf("rcs(%s, 4)", v),
        setdiff(vars, v),
        "SL"
      ),
      collapse = " + "
    )
  ))
  
  fit <- lrm(
    formula_rcs,
    data = dt,
    x = TRUE,
    y = TRUE
  )
  
  if (isTRUE(fit$fail)) {
    stop("模型拟合失败：", v)
  }
  
  a <- anova(fit)
  rownames(a) <- trimws(rownames(a))
  
  p_text <- paste0(
    "P for overall ", p_format(a[v, "P"]),
    "\nP for nonlinear ", p_format(a["Nonlinear", "P"])
  )
  
  # 完整观测范围；预测网格包含中位数
  x_grid <- sort(unique(c(
    seq(
      min(dt[[v]]),
      max(dt[[v]]),
      length.out = 500
    ),
    median(dt[[v]])
  )))
  
  pred <- as.data.frame(Predict(
    fit,
    factors = setNames(list(x_grid), v),
    ref.zero = TRUE,
    fun = exp,
    conf.int = 0.95
  ))
  
  pred$x <- pred[[v]]
  
  if (any(!is.finite(as.matrix(pred[c("yhat", "lower", "upper")])))) {
    stop("预测值或置信区间存在非有限值：", v)
  }
  
  y_top <- max(1, pred$upper) * 1.08
  
  # 直方图仅展示变量分布，柱高经过缩放，不对应OR数值
  h <- hist(dt[[v]], breaks = 25, plot = FALSE)
  
  hist_data <- data.frame(
    xmin = head(h$breaks, -1),
    xmax = tail(h$breaks, -1),
    height = h$counts / max(h$counts) * y_top * 0.88
  )
  
  ggplot() +
    geom_rect(
      data = hist_data,
      aes(
        xmin = xmin,
        xmax = xmax,
        ymin = 0,
        ymax = height
      ),
      fill = "#B0D5DF",
      alpha = 0.55,
      color = "#73858B",
      linewidth = 0.25
    ) +
    geom_ribbon(
      data = pred,
      aes(x = x, ymin = lower, ymax = upper),
      fill = "#F39B7F",
      alpha = 0.30
    ) +
    geom_hline(
      yintercept = 1,
      color = "#858585",
      linetype = "dashed",
      linewidth = 0.35
    ) +
    geom_line(
      data = pred,
      aes(x = x, y = yhat),
      color = "#DE876E",
      linewidth = 0.70
    ) +
    annotate(
      "text",
      x = Inf,
      y = Inf,
      label = p_text,
      hjust = 1.08,
      vjust = 1.25,
      family = "serif",
      size = 2.35,
      lineheight = 1.15
    ) +
    scale_x_continuous(
      breaks = scales::breaks_pretty(n = 4),
      expand = expansion(mult = c(0.01, 0.02))
    ) +
    scale_y_continuous(
      breaks = scales::breaks_pretty(n = 4),
      expand = expansion(mult = c(0, 0.02))
    ) +
    coord_cartesian(
      xlim = range(x_grid),
      ylim = c(0, y_top)
    ) +
    labs(
      x = axis_labels[[v]],
      y = "Adjusted OR (95% CI)"
    ) +
    theme_classic(base_size = 8, base_family = "serif") +
    theme(
      aspect.ratio = 0.85,
      text = element_text(color = "#222222"),
      axis.title.x = element_text(
        size = 8,
        lineheight = 0.95,
        margin = margin(t = 4)
      ),
      axis.title.y = element_text(
        size = 8,
        margin = margin(r = 3)
      ),
      axis.text = element_text(
        size = 7,
        color = "#333333"
      ),
      axis.line = element_blank(),
      axis.ticks = element_line(linewidth = 0.3),
      axis.ticks.length = grid::unit(1.2, "mm"),
      panel.border = element_rect(
        color = "#888888",
        fill = NA,
        linewidth = 0.35
      ),
      plot.margin = margin(t = 7, r = 5, b = 3, l = 5)
    )
})

options(datadist = old_datadist)

# ============================================================
# 5. 组合排版：上排3图，下排2图居中
# ============================================================

aligned <- cowplot::align_plots(
  plotlist = plots,
  align = "hv",
  axis = "tblr"
)

positions <- data.frame(
  x = c(0, 1 / 3, 2 / 3, 1 / 6, 1 / 2),
  y = c(0.5, 0.5, 0.5, 0, 0)
)

figure_rcs <- cowplot::ggdraw()

for (i in seq_along(aligned)) {
  figure_rcs <- figure_rcs +
    cowplot::draw_plot(
      aligned[[i]],
      x = positions$x[i],
      y = positions$y[i],
      width = 1 / 3,
      height = 0.5
    )
}

figure_rcs <- figure_rcs +
  cowplot::draw_plot_label(
    label = LETTERS[1:5],
    x = positions$x + 0.005,
    y = positions$y + 0.498,
    hjust = 0,
    vjust = 1,
    size = 10,
    fontfamily = "serif",
    fontface = "bold"
  ) +
  theme(
    plot.background = element_rect(
      fill = "white",
      color = NA
    )
  )

# ============================================================
# 6. 仅输出RCS图：矢量PDF及600 dpi TIFF
# ============================================================

ggsave(
  filename = file.path(output_dir, "Figure_RCS_5variables.pdf"),
  plot = figure_rcs,
  device = grDevices::cairo_pdf,
  width = 190,
  height = 125,
  units = "mm",
  bg = "white"
)

ggsave(
  filename = file.path(output_dir, "Figure_RCS_5variables.tiff"),
  plot = figure_rcs,
  device = "tiff",
  width = 190,
  height = 125,
  units = "mm",
  dpi = 600,
  compression = "lzw",
  bg = "white"
)

print(figure_rcs)