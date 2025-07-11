# Car Price Prediction using Multiple Linear Regression (Interpretability)

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

Several transformations and modeling decisions were made to improve linear model assumptions and multicollinearity (after many iterations of the project):

- **Engine Type**:  
  The original `enginetype` variable had 6 categories, but some of them had very few observations, which made them unreliable for statistical modeling. As shown in the bar chart [`graficos/engine_type_distribution.png`](graficos/engine_price_todos.png), the data was highly imbalanced. To address this and improve model stability, we grouped the engine types into a binary feature: `engine_class_performance`, representing *economy* vs *performance* engines.  
Both versions—6-class and 2-class—showed statistically significant effects on price according to ANOVA, confirming that the simplified version retained meaningful explanatory power (F = 41.33, p < 0.001).

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
