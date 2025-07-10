library(httpgd)
hgd()
library(ggplot2)
library(dplyr)
library(e1071)
library(readr)
library(dplyr)
library(GGally)
library(corrplot)
library(MASS)
library(ppcor)
library(car)
library(nortest)
library(lmtest)

df <- read.csv("clases hamdi/tarearl/datasets/CarPrice_Assignment_Clean.csv", row.names='car_ID')
df$fueltype_gas <- as.numeric(df$fueltype_gas == "True")
df$engine_class_performance <- as.numeric(df$engine_class_performance == "True")
df$price <- log(df$price)
df$curbweight <- log(df$curbweight)
df$power_efficiency <- df$horsepower / df$citympg
glimpse(df)
str(df)
dim(df)

vars_importantes <- c("price", "curbweight", "enginesize", "power_efficiency", 
                        "carlength", "carwidth", "wheelbase", "boreratio",
                        "engine_class_performance", "fueltype_gas","symboling"
                        )

cor_kendall <- cor(df[ , vars_importantes], method = "kendall", use = "pairwise.complete.obs")

# Guardar en archivo PNG
png("clases hamdi/tarearl/graficos/corrplot.png",
    width = 4000, height = 4000, res = 400)

# Corrplot con método mixto
corrplot.mixed(cor_kendall,
               upper = 'pie',
               lower = 'shade',
               order = "hclust",
               tl.col = "black",
               tl.srt = 30,
               tl.cex = 0.8,         # Tamaño de texto más pequeño
               addCoef.col = "black")
dev.off()
# Primer modelo de prueba con todas las variables
fit_all <- lm(price ~ ., data = df[, vars_importantes])
summary(fit_all)
car::vif(fit_all)
# Correlación parcial
pc_all <- pcor(df[, vars_importantes])          # matriz de correlaciones parciales
round(pc_all$estimate, 2)
round(pc_all$p.value , 3)
mat_corr_parcial <- round(pc_all$estimate, 2)
png("clases hamdi/tarearl/graficos/correlacion_parcial.png", width = 2000, height = 2000, res = 300)
corrplot(mat_corr_parcial,
         method = "color",         # o "circle", "number", "shade", etc.
         type = "upper",
         order = "hclust",
         tl.col = "black",
         tl.cex = 0.8,
         addCoef.col = "black",   # Mostrar los valores en el gráfico
         number.cex = 0.7)

dev.off()

# .... LASSO con todo ----- #
library(glmnet)
vars_importantes <- unique(c("price", vars_importantes))
df <- df[, vars_importantes]
X <- model.matrix(price ~ ., data = df)[, -1]
y <- df$price
cv_lasso <- cv.glmnet(X, y, alpha = 1)
coef(cv_lasso, s = "lambda.min")
# Decidimos quedarnos con curbweight como variable de tamaño

# Alta colinealidad, eliminamos variables de tamaño y dejamos solo curbweight
vars_quedan <- c("price","symboling","curbweight", "power_efficiency", 
                         "boreratio", "engine_class_performance", "fueltype_gas", "enginesize"
                        )
df <- df[, vars_quedan]
summary(df)
# Entrenamiento del modelo LM
set.seed(49)
train.filas <- sample(nrow(df), nrow(df) * 0.8,replace = FALSE)
train.datos <- df[train.filas, ]
test.datos <- df[-train.filas, ]
summary(train.datos)

fit <- lm(price ~ curbweight  + power_efficiency + enginesize + boreratio + engine_class_performance + fueltype_gas, data = train.datos)
fit_base <- lm(price ~ 1, data = train.datos)
scope_forward <- formula(~ curbweight + power_efficiency+ enginesize +
                         boreratio + engine_class_performance + fueltype_gas)
step_aic <- stepAIC(fit,  direction="both")
n<-nrow(df)
step_bic <- stepAIC(fit, direction="both", k=log(n))
forwAIC <- stepAIC(fit_base,
                   scope = list(lower = ~1, upper = scope_forward),
                   direction = "both")
forwBIC <- stepAIC(fit_base,
                   scope = list(lower = ~1, upper = scope_forward),
                   direction = "both", k = log(n))
summary(step_aic)
summary(step_bic)
summary(forwAIC)
summary(forwBIC)


# Aplicar los dos ajustes para evaluar como se 
# desarrollan

fitAIC <- lm(formula(step_aic), data = train.datos)
fitBIC <- lm(formula(step_bic), data = train.datos)
summary(fitAIC)
summary(fitBIC)
# Quitamos engine_class_performance porque no hay suficiente
# evidencia para decir que es un buen predictor y no reduce demasiado r2
# además variables como horsepower contienen información
#Colinealidad:problema de colinealidad con el ajuste
car::vif(fitBIC)
car::vif(fitAIC)


# --- No hay problemas de colinealidad ---
fit_lasso <- lm(price ~ curbweight+ power_efficiency + fueltype_gas+symboling, data = train.datos)
summary(fit_lasso)
vif(fit_lasso)
qqPlot(residuals(fit_lasso))
library(nortest)
ad.test(fit_lasso$residuals)
# 	•	LASSO detectó colinealidad y decidió mantener solo una combinación representativa: por ejemplo, eligió curbweight y citympg (los más informativos).
#	•	horsepower y enginesize, que estaban correlacionados con curbweight, fueron casi eliminados (coef ≈ 0).
#	•	fueltype_gas quedó con un coeficiente pequeño pero no nulo.
# -------


#####
#Verificación de ciertos puntos de metodología que vamos a estudiar más en detalle en adelante
#####

# El modelo lasso esta ahí, podría ser útil para predicción,
# pero por interpretabilidad nos mantendremos con BIC.
# ------ MODELO FINAL ------
fit_lm <- lm(price ~ curbweight + power_efficiency+ fueltype_gas, data = train.datos)
summary(fit_lm)
# Outliers
par(mfrow = c(1, 1))
png("clases hamdi/tarearl/graficos/influencePlot_fitlasso.png", width = 800, height = 600)
influencePlot(fit_lm)
title("Influence Plot - LM")
dev.off()
# Quitamos los outliers que pueden influir
influencePlot(fit_lm)
title("Influence Plot - Modelo Lasso")



# RLM para los outliers, ya que no queremos deshacernos de ellos por poder ser importantes
# poca probabilidad de que sean errores de medición
# Comparación BIC vs RLM


#   fit_rlm <- rlm(price ~ curbweight + horsepower + citympg + fueltype_gas, 
#                    data = train.datos)
fit_rlm <- rlm(price ~ curbweight + power_efficiency +fueltype_gas,  
            data = train.datos)
summary(fit_rlm)


# Recomendación Hacer bootstrap
# Residuos estandarizados
resid_lm  <- residuals(fit_lm)
resid_rlm <- residuals(fit_rlm)# QQ Plot LM
png("clases hamdi/tarearl/graficos/qqplots_comparacion.png", width = 1200, height = 1000)
# Panel con 2 gráficos lado a lado
par(mfrow = c(1, 2))
qqPlot(resid_lm, main = "QQ Plot - LM", col.lines = "blue")
qqPlot(resid_rlm, main = "QQ Plot - RLM", col.lines = "darkred")
dev.off()
par(mfrow = c(1, 2))
qqPlot(resid_lm, main = "QQ Plot - LM", col.lines = "blue")
qqPlot(resid_rlm, main = "QQ Plot - RLM", col.lines = "darkred")
kurtosis(fit_rlm$residuals)
skewness(fit_rlm$residuals)
#Test de normalidad
ad.test(fit_rlm$residuals)
ad.test(fit_lm$residuals)
cvm.test(fit_rlm$residuals)
lillie.test(fit_rlm$residuals)
# Heterocedasticidad y autocorrelación
dwtest(fit_rlm)
bptest(fit_rlm)
 
# Spread-level plot
par(mfrow=c(1,1))
png('clases hamdi/tarearl/graficos/residuals_vs_fitted_rlm.png',width = 800, height = 600)
spreadLevelPlot(fit_rlm, robust.line = TRUE)
dev.off()
spreadLevelPlot(fit_rlm, robust.line = TRUE)

skewness(fit_rlm$residuals)
kurtosis(fit_rlm$residuals)

# ------ BOOTSTRAP -------------- #
library(boot)
boot_rlm <- function(data, indices) {
  # Re-muestreo de los datos
  d <- data[indices, ]
  
  modelo <- rlm(price ~ curbweight + power_efficiency +fueltype_gas, data = d,
            maxit = 50)
  
  return(coef(modelo))
}

boot_result <- boot(data = train.datos, statistic = boot_rlm, R = 1000)
# Intervalos de Confianza
# Para el coeficiente de curbweight (index = 2)
boot.ci(boot_result, type = "perc", index = 2)

# Para fueltype_gas (index = 3)
boot.ci(boot_result, type = "perc", index = 3)

# Para power_efficiency (index = 4)
boot.ci(boot_result, type = "perc", index = 4)

hist(boot_result$t[,2], main = "Distribución bootstrap del coeficiente curbweight", xlab = "Valor coeficiente")
# Comparativa 
pred_lm  <- predict(fit_lm,  newdata = test.datos)
pred_rlm <- predict(fit_rlm, newdata = test.datos)
obs <- test.datos$price
# RMSE
rmse_lm  <- sqrt(mean((test.datos$price - pred_lm)^2))
rmse_rlm <- sqrt(mean((test.datos$price - pred_rlm)^2))
# MAE
mae_lm  <- mean(abs(test.datos$price - pred_lm))
mae_rlm <- mean(abs(test.datos$price - pred_rlm))
# R² manual (opcional)
r2_lm  <- 1 - sum((test.datos$price - pred_lm)^2) / sum((test.datos$price - mean(test.datos$price))^2)
r2_rlm <- 1 - sum((test.datos$price - pred_rlm)^2) / sum((test.datos$price - mean(test.datos$price))^2)
cat("🔍 Comparación de modelos:\n")
cat("LM  - RMSE:", round(rmse_lm, 4), "| MAE:", round(mae_lm, 4), "| R²:", round(r2_lm, 4), "\n")
cat("RLM - RMSE:", round(rmse_rlm, 4), "| MAE:", round(mae_rlm, 4), "| R²:", round(r2_rlm, 4), "\n")

# -------------------------
# FUNCIÓN DE TRANSFORMACIÓN
# -------------------------

predict_rlm_real <- function(fit_rlm, newdata) {
  pred_log <- predict(fit_rlm, newdata = newdata)
  residuos <- fit_rlm$residuals
  correction_factor <- mean(exp(residuos))
  pred_real <- exp(pred_log) * correction_factor
  return(list(pred_log = pred_log,
              pred_real = pred_real,
              correction_factor = correction_factor))
}

# -------------------------
# APLICACIÓN A TEST SET
# -------------------------

predicciones <- predict_rlm_real(fit_rlm, test.datos)

# Predicciones
pred_log <- predicciones$pred_log
pred_real <- predicciones$pred_real

# Valores reales
obs_log <- test.datos$price
obs_real <- exp(test.datos$price)

# Errores
error_log <- obs_log - pred_log
error_real <- obs_real - pred_real

# -------------------------
# GRAFICO 1: PREDICCIONES
# -------------------------

par(mfrow = c(1, 2), oma = c(0, 0, 2, 0))

# (A) Predicción en escala log
plot(obs_log, pred_log,
     xlab = "Log(Precio) real",
     ylab = "Log(Precio) predicho",
     main = "Escala logarítmica",
     pch = 19, col = "steelblue")
abline(0, 1, col = "red", lwd = 2)

# (B) Predicción en escala real
plot(obs_real, pred_real,
     xlab = "Precio real ($)",
     ylab = "Precio predicho ($)",
     main = "Escala real corregida",
     pch = 19, col = "forestgreen")
abline(0, 1, col = "red", lwd = 2)

mtext("Predicción vs Observado (RLM)", outer = TRUE, cex = 1.4)

# -------------------------
# GRAFICO 2: ERRORES
# -------------------------

par(mfrow = c(1, 2), oma = c(0, 0, 2, 0))

# (A) Error en escala log
plot(obs_log, error_log,
     xlab = "Log(Precio) real",
     ylab = "Error de predicción (log)",
     main = "Errores escala logarítmica",
     pch = 19, col = "steelblue")
abline(h = 0, col = "gray40", lwd = 2)

# (B) Error en escala real
plot(obs_real, error_real,
     xlab = "Precio real ($)",
     ylab = "Error de predicción ($)",
     main = "Errores escala real corregida",
     pch = 19, col = "tomato")
abline(h = 0, col = "gray40", lwd = 2)

mtext("Errores de predicción (RLM)", outer = TRUE, cex = 1.4)

df_pred <- data.frame(obs_real, pred_real, error_real)

ggplot(df_pred, aes(x = obs_real, y = pred_real, fill = error_real)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                       midpoint = 0, name = "Error ($)") +
  labs(x = "Precio real", y = "Precio predicho",
       title = "Heatmap del error de predicción (escala real)") +
  theme_minimal()

  # ---- Paneles combinados ------ #
png('clases hamdi/tarearl/graficos/prediccionescombinadas.png',width = 800, height = 600)
par(mfrow = c(2, 2))

# (1) log-log
plot(obs_log, pred_log, main = "Predicción log",
     xlab = "Log(real)", ylab = "Log(pred)", pch = 19, col = "steelblue")
abline(0, 1, col = "red")

# (2) real-real
plot(obs_real, pred_real, main = "Predicción real",
     xlab = "Real ($)", ylab = "Predicho ($)", pch = 19, col = "forestgreen")
abline(0, 1, col = "red")

# (3) error-log
plot(obs_log, error_log, main = "Error log",
     xlab = "Log(real)", ylab = "Error", pch = 19, col = "gray")
abline(h = 0, col = "red")

# (4) error-real
plot(obs_real, error_real, main = "Error real",
     xlab = "Real ($)", ylab = "Error ($)", pch = 19, col = "tomato")
abline(h = 0, col = "red")
dev.off()
par(mfrow = c(1,1))
# ----- Error Absoluto vs Precio ------

plot(obs_real, abs(error_real),
     xlab = "Precio real ($)", ylab = "|Error| ($)",
     main = "Error absoluto vs Precio",
     pch = 19, col = "purple")


#### ------------ PREDICCIONES DE NIVEL GENERAL ---------------------- #####
# BOOTSTRAP
boot_pred <- function(data, indices, new_list){
  d   <- data[indices, ]
  fit <- rlm(price ~ curbweight + power_efficiency + fueltype_gas, data = d)

  sapply(new_list, function(df) as.numeric(predict(fit, df)))
}

perfil <- data.frame(
  curbweight             = median(train.datos$curbweight),
  power_efficiency       = median(train.datos$power_efficiency),
  fueltype_gas           = 1
)
# Alfa-Romeo Giulia
#car_ID
     # 1    111
     # Name: horsepower, dtype: int64
     # car_ID
     # 1    21
#    Name: citympg, dtype: int64
individuo <- data.frame(
  curbweight             = log(2548),
  power_efficiency       = 111/21,
  fueltype_gas           = 1
)

boot_both <- boot(
  data      = train.datos,
  statistic = function(data, i) boot_pred(data, i, list(perfil, individuo)),
  R         = 1000
)

boot.ci(boot_both, type = "perc", index = 1)  # IC del perfil
boot.ci(boot_both, type = "perc", index = 2)  # IC del auto real

# Predicciones puntuales (con corrección de sesgo)
pred_perfil_real     <- predict_rlm_real(fit_rlm, perfil)$pred_real
pred_individuo_real  <- predict_rlm_real(fit_rlm, individuo)$pred_real

# IC de predicción en escala real
boot_real_perfil    <- exp(boot_both$t[, 1])
boot_real_individuo <- exp(boot_both$t[, 2])
boot_both$t[, 2]

IC_perfil    <- quantile(boot_real_perfil,    c(0.025, 0.975))
IC_individuo <- quantile(boot_real_individuo, c(0.025, 0.975))

# Reporte
cat("Pred PERFIL:", round(pred_perfil_real, 0),
    "\nIC 95%:", round(IC_perfil[1], 0), "-", round(IC_perfil[2], 0), "\n")

cat("Pred INDIVIDUO:", round(pred_individuo_real, 0),
    "\nIC 95%:", round(IC_individuo[1], 0), "-", round(IC_individuo[2], 0), "\n")
