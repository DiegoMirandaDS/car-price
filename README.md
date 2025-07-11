# Car Price Prediction using Robust Linear Regression

This project aims to model and predict car prices using robust multiple linear regression, emphasizing model interpretability and assumption diagnostics. The final model is able to estimate car prices with confidence intervals using bootstrap.

---

## 📁 Dataset

The dataset was obtained from UCI Irvine Machine Learning Repository and contains detailed specifications (engine, body, dimensions, fuel, etc.) for 205 cars.

- **Original source**: [Automobile](https://archive.ics.uci.edu/dataset/10/automobile)

---

## 👤 Author

**Diego Miranda**  
[GitHub Profile](https://github.com/DiegoMirandaDS)

---

## ⚙️ Feature Engineering

Several transformations and modeling decisions were made to improve linear model assumptions and multicollinearity:

- **Engine Type**:  
  The original `enginetype` variable had 6 categories. After an ANOVA test showed a statistically significant difference (F = 19.29, p < 0.001), we grouped them into a binary feature: `engine_class_performance`, representing economy vs performance-oriented engines.

- **Fuel Type**:  
  Despite a non-significant ANOVA result for `fueltype` (p > 0.05), it was retained in the model due to its domain relevance and its ability to slightly increase explained variance and interpretability.

- **Power Efficiency**:  
  A new feature, `power_efficiency = horsepower / citympg`, was created to capture the tradeoff between performance and efficiency. This helped reduce multicollinearity while improving predictive performance.

- **Transformations**:
  - Target variable `price` was log-transformed to improve residual normality.
  - `curbweight` was also log-transformed.
  - Standard correlation (`kendall`) and **partial correlation** matrices were used to evaluate feature redundancy.  
    📊 Plots: [`graficos/corrplot.png`](graficos/corrplot.png), [`graficos/partial_correlations.png`](graficos/partial_correlations.png)

---

## 🧪 Modeling Process

### 🔍 Model Selection

- All relevant variables were first included in a full model.
- Dimensionality was reduced using:
  - **Stepwise AIC/BIC**
  - **LASSO regression** (via `glmnet`)
- Selected features for final model:
