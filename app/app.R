# ==============================================================================
# ESWL Treatment Outcome Prediction - Shiny App (v4 with SHAP)
# Model: XGBoost (6 preoperative predictors)
# ==============================================================================

suppressPackageStartupMessages({
  library(shiny)
  library(tidymodels)
  library(workflows)
  library(recipes)
  library(parsnip)
  library(xgboost)
  library(dplyr)
  library(ggplot2)
})

# ---------- Configuration ----------
MODEL_PATH <- Sys.getenv("ESWL_MODEL_PATH", unset = "04_XGBoost_best_CV_refit_621.rds")
THRESHOLD <- 0.7307063937187195

# ---------- Load model ----------
cat("Loading model...\n")
model <- readRDS(MODEL_PATH)
stopifnot(inherits(model, "workflow"))
stopifnot(isTRUE(model$trained))
cat(sprintf("Model loaded! Engine: %s\n", model$fit$actions$model$spec$engine))

# ---------- Extract components for SHAP ----------
cat("Extracting recipe and engine for SHAP...\n")
recipe_obj <- extract_recipe(model)
xgb_engine <- extract_fit_engine(model)
model_features <- xgb_engine$feature_names
cat(sprintf("Model features: %d\n", length(model_features)))

# ---------- Calculate global variable importance ----------
cat("Calculating variable importance...\n")
var_imp_raw <- xgb.importance(model = xgb_engine)

var_importance <- var_imp_raw %>%
  mutate(Variable = case_when(
    grepl("Stone_volume", Feature) ~ "Stone volume",
    grepl("Mean_CT_attenuation", Feature) ~ "Mean CT attenuation",
    grepl("Ureteral_wall_thickness", Feature) ~ "Ureteral wall thickness",
    grepl("Hydronephrosis_measure", Feature) ~ "Hydronephrosis",
    grepl("Subcutaneous_fat_thickness", Feature) ~ "Subcutaneous fat thickness",
    grepl("Stone_location", Feature) ~ "Stone location",
    TRUE ~ Feature
  )) %>%
  group_by(Variable) %>%
  summarise(Gain = sum(Gain), .groups = "drop") %>%
  arrange(desc(Gain)) %>%
  mutate(
    Gain_pct = round(Gain / sum(Gain) * 100, 1),
    Bar_width = paste0(Gain_pct, "%")
  )

# ---------- Prediction function ----------
predict_eswl <- function(input_list) {
  new_data <- data.frame(
    Case_ID = 99999,
    Stone_volume = input_list$Stone_volume,
    Mean_CT_attenuation = input_list$Mean_CT_attenuation,
    Ureteral_wall_thickness = input_list$Ureteral_wall_thickness,
    Hydronephrosis_measure = input_list$Hydronephrosis_measure,
    Subcutaneous_fat_thickness = input_list$Subcutaneous_fat_thickness,
    Stone_location = input_list$Stone_location,
    check.names = FALSE
  )
  pred <- predict(model, new_data = new_data, type = "prob")
  prob_success <- pred$.pred_1
  list(
    prob = prob_success,
    class = ifelse(prob_success >= THRESHOLD, "Success", "Failure"),
    risk = ifelse(prob_success >= 0.85, "High probability",
           ifelse(prob_success >= THRESHOLD, "Moderate probability", "Low probability")),
    new_data = new_data
  )
}

# ---------- SHAP calculation function ----------
calculate_shap <- function(new_data) {
  processed <- bake(recipe_obj, new_data = new_data)
  available <- intersect(model_features, names(processed))
  if (length(available) != length(model_features)) {
    return(NULL)
  }
  dmatrix <- xgb.DMatrix(data = as.matrix(processed[, model_features]))
  shap_matrix <- predict(xgb_engine, dmatrix, predcontrib = TRUE)

  shap_df <- data.frame(
    Feature = colnames(shap_matrix)[-ncol(shap_matrix)],
    SHAP = as.numeric(shap_matrix[1, -ncol(shap_matrix)])
  ) %>%
    mutate(Variable = case_when(
      grepl("Stone_volume", Feature) ~ "Stone volume",
      grepl("Mean_CT_attenuation", Feature) ~ "Mean CT attenuation",
      grepl("Ureteral_wall_thickness", Feature) ~ "Ureteral wall thickness",
      grepl("Hydronephrosis_measure", Feature) ~ "Hydronephrosis",
      grepl("Subcutaneous_fat_thickness", Feature) ~ "Subcutaneous fat thickness",
      grepl("Stone_location", Feature) ~ "Stone location",
      TRUE ~ Feature
    )) %>%
    group_by(Variable) %>%
    summarise(SHAP = sum(SHAP), .groups = "drop") %>%
    arrange(desc(abs(SHAP)))

  shap_df
}

# ---------- SHAP waterfall plot ----------
plot_shap_waterfall <- function(shap_df, prob) {
  shap_df$Variable <- factor(shap_df$Variable, levels = rev(shap_df$Variable))
  shap_df$Direction <- ifelse(shap_df$SHAP >= 0,
                               "Increases success probability",
                               "Decreases success probability")

  ggplot(shap_df, aes(x = Variable, y = SHAP, fill = Direction)) +
    geom_col(width = 0.65) +
    geom_hline(yintercept = 0, color = "#666666", linewidth = 0.4) +
    coord_flip() +
    scale_fill_manual(values = c("Increases success probability" = "#1565C0",
                                  "Decreases success probability" = "#E53935")) +
    scale_y_continuous(labels = function(x) sprintf("%.2f", x)) +
    labs(
      title = "Prediction Breakdown (SHAP values)",
      subtitle = sprintf("Predicted success probability: %.1f%%  |  Blue = pushes probability up, Red = pushes it down", prob * 100),
      x = NULL,
      y = "SHAP value (contribution in log-odds)"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = 10),
      plot.title = element_text(face = "bold", size = 13, color = "#0D47A1"),
      plot.subtitle = element_text(size = 9.5, color = "#666666"),
      axis.text.y = element_text(size = 10, face = "bold", color = "#333333"),
      axis.text.x = element_text(size = 9, color = "#666666"),
      axis.title.x = element_text(size = 9.5, color = "#555555"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = margin(10, 12, 6, 6)
    )
}

# ---------- UI ----------
ui <- fluidPage(
  tags$style(HTML("
    body {
      background-color: #EBF2FA;
      color: #0D2B4E;
      font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
    }
    .title-panel {
      background: linear-gradient(135deg, #0D47A1 0%, #1565C0 50%, #1E88E5 100%);
      color: white;
      padding: 24px 32px;
      border-radius: 0 0 16px 16px;
      margin-bottom: 24px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .title-panel h1 { margin: 0; font-size: 26px; font-weight: 700; }
    .title-panel p { margin: 6px 0 0 0; font-size: 14px; opacity: 0.9; }
    .input-card {
      background: white;
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      margin-bottom: 16px;
    }
    .input-card h3 {
      color: #0D47A1;
      font-size: 16px;
      font-weight: 600;
      margin: 0 0 16px 0;
      padding-bottom: 8px;
      border-bottom: 2px solid #E3F2FD;
    }
    .result-card {
      background: white;
      border-radius: 12px;
      padding: 28px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      margin-bottom: 16px;
      text-align: center;
    }
    .prob-big {
      font-size: 56px;
      font-weight: 800;
      line-height: 1;
      margin: 12px 0;
      color: #0D47A1;
    }
    .prob-label { font-size: 14px; color: #666; margin-bottom: 4px; }
    .class-badge {
      display: inline-block;
      padding: 8px 28px;
      border-radius: 24px;
      font-size: 20px;
      font-weight: 700;
      margin: 12px 0;
    }
    .class-success { background: #E3F2FD; color: #0D47A1; border: 2px solid #42A5F5; }
    .class-failure { background: #FFEBEE; color: #B71C1C; border: 2px solid #EF5350; }
    .risk-tag {
      display: inline-block;
      padding: 4px 14px;
      border-radius: 12px;
      font-size: 13px;
      font-weight: 600;
      margin-top: 8px;
    }
    .risk-high { background: #BBDEFB; color: #0D47A1; }
    .risk-mid  { background: #FFF9C4; color: #F57F17; }
    .risk-low  { background: #FFCDD2; color: #B71C1C; }
    .progress-bar-container {
      width: 100%;
      height: 24px;
      background: #E0E0E0;
      border-radius: 12px;
      overflow: hidden;
      margin: 16px 0;
      position: relative;
    }
    .progress-bar-fill {
      height: 100%;
      border-radius: 12px;
      transition: width 0.6s ease;
      background: linear-gradient(90deg, #64B5F6, #1565C0);
    }
    .progress-bar-text {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      font-size: 12px;
      font-weight: 600;
      color: #333;
    }
    .varimp-bar-container {
      width: 100%;
      height: 18px;
      background: #E3F2FD;
      border-radius: 4px;
      overflow: hidden;
      position: relative;
    }
    .varimp-bar-fill {
      height: 100%;
      border-radius: 4px;
      background: linear-gradient(90deg, #42A5F5, #1565C0);
      transition: width 0.6s ease;
    }
    .varimp-table td { vertical-align: middle !important; }
    .varimp-name { font-weight: 600; color: #0D47A1; font-size: 13px; }
    .varimp-pct { font-weight: 700; color: #1565C0; font-size: 13px; text-align: right; }
    .btn-predict {
      background: linear-gradient(135deg, #1565C0, #1E88E5) !important;
      color: white !important;
      font-weight: 600 !important;
      font-size: 16px !important;
      padding: 12px 32px !important;
      border-radius: 8px !important;
      border: none !important;
      box-shadow: 0 2px 8px rgba(21,101,192,0.3) !important;
      width: 100%;
    }
    .btn-predict:hover {
      background: linear-gradient(135deg, #0D47A1, #1565C0) !important;
      box-shadow: 0 4px 12px rgba(21,101,192,0.4) !important;
    }
    .btn-reset {
      background: white !important;
      color: #666 !important;
      border: 1px solid #DDD !important;
      border-radius: 8px !important;
      padding: 8px 20px !important;
      width: 100%;
      margin-top: 8px;
    }
    .slider-label { font-weight: 600; color: #333; font-size: 13px; }
    .slider-unit { color: #999; font-size: 11px; font-weight: 400; }
    .footer-note {
      text-align: center;
      color: #999;
      font-size: 12px;
      margin-top: 24px;
      padding: 16px;
    }
    .threshold-note {
      background: #FFF8E1;
      border-left: 4px solid #FFC107;
      padding: 10px 14px;
      border-radius: 0 6px 6px 0;
      font-size: 12px;
      color: #795548;
      margin-top: 12px;
    }
    .shap-placeholder {
      text-align: center;
      color: #999;
      font-size: 13px;
      padding: 40px 20px;
    }
    .form-control { font-size: 13px; }
    .selectize-input { font-size: 13px; }
    .measure-hint {
      font-size: 11px; color: #888; margin-top: -10px;
      margin-bottom: 12px; line-height: 1.35; padding-left: 2px;
    }
    .measure-guide-panel {
      background: #F5F9FF; border: 1px solid #BBDEFB;
      border-radius: 8px; padding: 12px 14px; margin-top: 8px;
      margin-bottom: 12px; font-size: 11.5px; line-height: 1.5; color: #333;
    }
    .measure-guide-panel p { margin: 4px 0; }
    .measure-guide-panel strong { color: #0D47A1; }
  ")),
  tags$script(HTML("
    Shiny.addCustomMessageHandler('update_progress', function(value) {
      var fill = document.getElementById('progress_fill');
      if (fill) { fill.style.width = value + '%'; }
    });
  ")),

  # Title
  div(class = "title-panel",
      h1("ESWL Treatment Outcome Prediction"),
      p("XGBoost Model \u00b7 6 Preoperative Features \u00b7 SHAP Explainability \u00b7 Extracorporeal Shock Wave Lithotripsy")
  ),

  fluidRow(
    # ---------- Left: Input Panel ----------
    column(width = 5,
      div(class = "input-card",
          h3("Patient Preoperative Features"),

          div(style = "margin-bottom: 14px;",
              div(class = "slider-label", "Stone location",
                  tags$span(class = "slider-unit", " (categorical)")),
              selectInput("Stone_location", NULL,
                          choices = c("Distal ureter" = "Distal ureter",
                                      "Mid ureter" = "Mid ureter",
                                      "Proximal ureter" = "Proximal ureter"),
                          selected = "Distal ureter"),
              div(class = "measure-hint", "Proximal (above sacrum) \u00b7 Middle (sacroiliac joint) \u00b7 Distal (below sacroiliac joint)")
          ),

          sliderInput("Stone_volume",
                      label = div(span("Stone volume", class = "slider-label"),
                                  span(" mm\u00b3", class = "slider-unit")),
                      min = 0, max = 200, value = 35, step = 1),
          div(class = "measure-hint", "\u03c0/6 \u00d7 length \u00d7 width \u00d7 depth; length on sagittal/coronal MPR"),

          sliderInput("Mean_CT_attenuation",
                      label = div(span("Mean CT attenuation", class = "slider-label"),
                                  span(" HU", class = "slider-unit")),
                      min = 100, max = 1500, value = 273, step = 10),
          div(class = "measure-hint", "Mean HU in freehand ROI on largest axial slice \u00b7 soft-tissue window 350/50"),

          sliderInput("Ureteral_wall_thickness",
                      label = div(span("Ureteral wall thickness", class = "slider-label"),
                                  span(" mm", class = "slider-unit")),
                      min = 0.5, max = 5.0, value = 1.5, step = 0.1),
          div(class = "measure-hint", "Max wall thickness at stone level \u00b7 soft-tissue window 350/50"),

          sliderInput("Hydronephrosis_measure",
                      label = div(span("Hydronephrosis", class = "slider-label"),
                                  span(" mm", class = "slider-unit")),
                      min = 0, max = 30, value = 13, step = 0.5),
          div(class = "measure-hint", "Max anteroposterior diameter of the renal pelvis; caliper only"),

          sliderInput("Subcutaneous_fat_thickness",
                      label = div(span("Subcutaneous fat thickness", class = "slider-label"),
                                  span(" mm", class = "slider-unit")),
                      min = 5, max = 40, value = 18, step = 0.5),
          div(class = "measure-hint", "Skin surface to abdominal fascia; same slice and direction"),

          div(style = "margin-top: 12px; margin-bottom: 4px;",
              checkboxInput("show_measure_guide",
                            label = tags$span(icon("book-open"), " How to measure?",
                                              style = "font-size: 12px; font-weight: 600; color: #1565C0;"),
                            value = FALSE, width = "100%")
          ),
          conditionalPanel(
            condition = "input.show_measure_guide == true",
            div(class = "measure-guide-panel",
                p(strong("All measurements on 1-mm axial non-contrast CT.")),
                p(strong("Window:"), "All quantitative measurements at soft-tissue window 350/50 HU. Bone window (2000/500 HU) only to aid stone localization."),
                tags$hr(style = "border-color: #BBDEFB; margin: 8px 0;"),
                p(strong("1. Stone location:"), "Classify on axial images; use coronal/sagittal MPR if needed."),
                tags$ul(style = "margin: 2px 0 6px 18px; padding: 0; font-size: 11.5px;",
                  tags$li("Proximal: ureteropelvic junction to upper sacral border"),
                  tags$li("Middle: overlying the sacroiliac joint"),
                  tags$li("Distal: lower sacroiliac border to ureterovesical junction")
                ),
                p(strong("2. Stone volume (SV), mm\u00b3:"), "Ellipsoid formula: \u03c0/6 \u00d7 length \u00d7 width \u00d7 depth."),
                tags$ul(style = "margin: 2px 0 6px 18px; padding: 0; font-size: 11.5px;",
                  tags$li("Length: max diameter on sagittal or coronal MPR"),
                  tags$li("Width & depth: two perpendicular diameters on the axial slice with the largest stone cross-section")
                ),
                p(strong("3. Mean CT attenuation (MeCT), HU:"), "On the axial slice with the largest stone cross-section, draw a freehand ROI covering the largest possible intrastone area. Exclude adjacent urine, surrounding soft tissue, and edge pixels affected by partial-volume effect. Do not apply a fixed 1-mm inset. MeCT = mean HU of all pixels inside the ROI."),
                p(strong("4. Ureteral wall thickness (UWT), mm:"), "On the axial slice at or closest to the stone level, measure the maximum thickness of the affected (stone-bearing) ureteral wall. Identify the wall boundary between the ureteral lumen and periureteral tissue."),
                p(strong("5. Hydronephrosis (DH), mm:"), "On axial CT, locate the renal pelvis at its widest level and measure the maximum anteroposterior diameter with electronic calipers only. No ROI delineation."),
                p(strong("6. Subcutaneous fat thickness (SFT), mm:"), "On the axial slice at the stone level, along the measurement direction from skin to stone, measure from the skin surface (outer margin) to the abdominal fascia (inner margin of subcutaneous fat).")
            )
          ),

          div(class = "threshold-note",
              sprintf("Classification threshold (Youden index): %.3f \u00b7 Probability \u2265 threshold = Success", THRESHOLD))
      ),

      actionButton("predict_btn", "Predict", class = "btn-predict"),
      actionButton("reset_btn", "Reset to defaults", class = "btn-reset")
    ),

    # ---------- Right: Results ----------
    column(width = 7,
      # Main result
      div(class = "result-card",
          div(class = "prob-label", "Probability of Treatment Success"),
          div(class = "prob-big",
              textOutput("prob_text", inline = TRUE),
              span(style = "font-size: 24px; font-weight: 400; color: #666;", "%")
          ),

          div(class = "progress-bar-container",
              div(class = "progress-bar-fill", style = "width: 0%;", id = "progress_fill"),
              div(class = "progress-bar-text", textOutput("progress_label", inline = TRUE))
          ),

          uiOutput("class_badge"),
          uiOutput("risk_tag")
      ),

      # SHAP waterfall
      div(class = "input-card",
          h3("Prediction Breakdown (SHAP)"),
          uiOutput("shap_plot_ui")
      ),


    )
  ),

  div(class = "footer-note",
      "This tool is for research reference only and does not constitute clinical diagnosis. \u00b7 Treatment decisions should be made in consultation with a qualified physician.")
)

# ---------- Server ----------
server <- function(input, output, session) {

  result <- eventReactive(input$predict_btn, {
    req(input$Stone_location)
    input_list <- list(
      Stone_volume = input$Stone_volume,
      Mean_CT_attenuation = input$Mean_CT_attenuation,
      Ureteral_wall_thickness = input$Ureteral_wall_thickness,
      Hydronephrosis_measure = input$Hydronephrosis_measure,
      Subcutaneous_fat_thickness = input$Subcutaneous_fat_thickness,
      Stone_location = input$Stone_location
    )
    tryCatch(
      predict_eswl(input_list),
      error = function(e) {
        showNotification(paste("Prediction error:", e$message), type = "error")
        NULL
      }
    )
  })

  # SHAP values reactive
  shap_result <- eventReactive(input$predict_btn, {
    req(result())
    tryCatch(
      calculate_shap(result()$new_data),
      error = function(e) {
        showNotification(paste("SHAP calculation error:", e$message), type = "warning")
        NULL
      }
    )
  })

  output$prob_text <- renderText({
    req(result())
    sprintf("%.1f", result()$prob * 100)
  })

  observeEvent(result(), {
    req(result())
    prob_pct <- result()$prob * 100
    session$sendCustomMessage("update_progress", prob_pct)
  }, ignoreNULL = TRUE)

  output$progress_label <- renderText({
    req(result())
    sprintf("%.1f%% success probability", result()$prob * 100)
  })

  output$class_badge <- renderUI({
    req(result())
    cls <- result()$class
    if (cls == "Success") {
      div(class = "class-badge class-success", "Predicted: Treatment Success")
    } else {
      div(class = "class-badge class-failure", "Predicted: Treatment Failure")
    }
  })

  output$risk_tag <- renderUI({
    req(result())
    risk <- result()$risk
    cls <- ifelse(grepl("High", risk), "risk-high",
            ifelse(grepl("Moderate", risk), "risk-mid", "risk-low"))
    div(class = paste("risk-tag", cls), risk)
  })

  # SHAP waterfall plot
  output$shap_plot <- renderPlot({
    req(shap_result(), result())
    validate(need(!is.null(shap_result()), "SHAP values not available"))
    plot_shap_waterfall(shap_result(), result()$prob)
  }, res = 120, height = 320)

  output$shap_plot_ui <- renderUI({
    if (is.null(shap_result())) {
      div(class = "shap-placeholder",
          "Click \"Predict\" to see how each feature contributes to this patient's prediction.")
    } else {
      plotOutput("shap_plot", height = "320px")
    }
  })

  # Reset button
  observeEvent(input$reset_btn, {
    updateSelectInput(session, "Stone_location", selected = "Distal ureter")
    updateSliderInput(session, "Stone_volume", value = 35)
    updateSliderInput(session, "Mean_CT_attenuation", value = 273)
    updateSliderInput(session, "Ureteral_wall_thickness", value = 1.5)
    updateSliderInput(session, "Hydronephrosis_measure", value = 13)
    updateSliderInput(session, "Subcutaneous_fat_thickness", value = 18)
  })
}

# ---------- Run ----------
shinyApp(ui = ui, server = server)
