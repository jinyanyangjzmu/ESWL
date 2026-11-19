# First-session ESWL outcome prediction for ureteral stones

An R Shiny application that estimates the probability of treatment success after the first session of extracorporeal shock wave lithotripsy (ESWL) for ureteral stones. It applies a fitted XGBoost model to six preoperative CT-derived predictors and shows how each predictor contributes to the individual prediction.

The study defined success as residual fragments smaller than 4 mm, no repeat ESWL within 4 weeks, and uncomplicated early recovery. Consult the associated manuscript for the full outcome definition and model development and validation methods.

## Run locally

Place the following files in the same directory.

| File | Purpose |
| --- | --- |
| `app.R` | Shiny interface, prediction, and SHAP visualization |
| `04_XGBoost_best_CV_refit_621.rds` | Required fitted tidymodels workflow containing the preprocessing recipe and XGBoost model |
| `README.md` | Project documentation |

Install the R packages used by the application.

```r
install.packages(c(
  "shiny", "tidymodels", "workflows", "recipes",
  "parsnip", "xgboost", "dplyr", "ggplot2"
))
```

Open `app.R` in RStudio and click **Run App**, or run the following from an R session in the app directory. For a different working directory, pass the path to the app folder in place of `"."`.

```r
shiny::runApp(appDir = ".", launch.browser = TRUE)
```

If startup fails, check that the model filename matches exactly and that it is alongside `app.R`. For model loading or prediction errors related to package compatibility, use the package versions from the environment that produced the RDS file. [Shiny runApp documentation](https://shiny.posit.co/r/reference/shiny/latest/runapp.html)

## Inputs

Enter the CT measurements manually. Expand **How to measure?** in the app to see the measurement protocol. The ranges below describe the app controls.

| Predictor | Unit | Input range or choices |
| --- | --- | --- |
| Stone location | Category | Proximal, middle, or distal ureter |
| Stone volume | mm³ | 0–200 |
| Mean CT attenuation | HU | 100–1500 |
| Ureteral wall thickness | mm | 0.5–5.0 |
| Hydronephrosis, measured as the maximum anteroposterior diameter of the renal pelvis | mm | 0–30 |
| Subcutaneous fat thickness | mm | 5–40 |

## Outputs

Click **Predict** to display the estimated probability of first-session ESWL success and the corresponding classification. **Reset to defaults** restores the example inputs.

- **Predicted class** uses a fixed threshold of **0.7307**. A success probability at or above the threshold is classified as **Success**. A probability below it is classified as **Failure**. The interface rounds the threshold to `0.731`.
- **Probability labels** are **Low** below 0.7307, **Moderate** from 0.7307 to below 0.85, and **High** at or above 0.85.
- **SHAP plot** shows the contribution of each predictor for the current case on the model's log-odds scale. Blue bars push the prediction toward success and red bars push it toward failure.

## GitHub and online deployment

For a repository dedicated to the app, place `README.md` and `app.R` at the repository root. Local execution also requires the matching model RDS. Before adding the RDS to a public repository, verify that the fitted workflow is cleared for public release and contains no patient-level records or identifiers. If the model is distributed separately, document how users can obtain the file.

GitHub hosts the source code. Browser-based use requires an R Shiny host, such as shinyapps.io or a server running Shiny Server. GitHub Pages can link to the hosted app from a static project page. [GitHub file upload guide](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository) · [GitHub Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)

## Intended use

This application is for research reference. Clinical decisions require assessment by a qualified clinician.
