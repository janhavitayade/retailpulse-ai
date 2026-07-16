## Initial Data Understanding & EDA Findings

The Rossmann dataset contains historical daily sales records for 1,115 retail stores over a period of approximately 2.5 years.

### Dataset Overview

* **Train Dataset:** 1,017,209 rows and 9 columns
* **Store Dataset:** 1,115 rows and 10 columns
* **Date Range:** January 1, 2013 to July 31, 2015
* **Unique Dates:** 942 days
* **Target Variable:** Sales

### Data Quality Assessment

The train dataset contains no missing values. However, the store dataset contains missing values in competition- and promotion-related fields:

* CompetitionDistance
* CompetitionOpenSinceMonth
* CompetitionOpenSinceYear
* Promo2SinceWeek
* Promo2SinceYear
* PromoInterval

These missing values are likely business-related rather than data errors, indicating that a store may not have a nearby competitor or may not participate in the Promo2 campaign.

### Sales Distribution

Summary statistics for the Sales variable:

* Mean Sales: 5,773.82
* Median Sales: 5,744
* Minimum Sales: 0
* Maximum Sales: 41,551
* Standard Deviation: 3,849.93

The mean and median sales values are very close, suggesting that the overall sales distribution is reasonably balanced, although some high-sales days are present.

### Zero Sales Analysis

A total of 172,871 records have sales equal to zero.

Investigation of store status revealed:

* 172,817 zero-sales records occurred when the store was closed (Open = 0)
* Only 54 zero-sales records occurred when the store was open (Open = 1)

This indicates that the vast majority of zero sales are operationally driven by store closures rather than unusual customer demand patterns.

### Forecasting Perspective

Since the objective is to build and compare multiple forecasting models, the dataset provides sufficient historical coverage with 942 daily observations across more than two years. The data is expected to exhibit both trend and seasonal behavior, making it suitable for classical forecasting methods (Moving Average, Exponential Smoothing), statistical models (ARIMA, SARIMA, SARIMAX), and machine learning approaches (Random Forest, XGBoost, LightGBM, CatBoost).

The next step is to aggregate daily sales across all stores to create a single time-series representation for trend and seasonality analysis.

# Exploratory Data Analysis (EDA) Summary

This repository contains the exploratory data analysis and baseline findings for our retail sales forecasting model. The following matrix summarizes the key numerical evidence discovered during data exploration and translates these findings into actionable business interpretations.

## Analysis Matrix

| Analysis Area | Finding & Numerical Evidence | Business Interpretation |
| :--- | :--- | :--- |
| **Dataset Size** | **Train:** 1,017,209 rows × 9 columns<br>**Store:** 1,115 rows × 10 columns | Provides a sufficiently large data footprint for robust forecasting and machine learning modeling. |
| **Time Coverage** | Nearly 3 years of historical data<br>• Timeline: 2013-01-01 to 2015-07-31<br>• 942 unique days | Offers enough historical depth to effectively capture long-term trends and yearly seasonality. |
| **Sales Distribution** | Sales are reasonably centered<br>• Mean: 5,773.82 \| Median: 5,744<br>• Std Dev: 3,849.93 \| Max: 41,551 | The proximity of the mean and median indicates a relatively balanced distribution across active periods. |
| **Zero Sales Analysis** | **172,871** total rows with `Sales = 0`<br>• **172,817** of these occurred when `Open = 0` | Zero sales primarily reflect structural store closures rather than a natural lack of customer demand. |
| **Trend Analysis** | Long-term upward trend observed | Visible in daily aggregated sales series, indicating consistent organic business growth over time. |
| **Weekly Seasonality** | **Strong weekly patterns exist:**<br>• Mon: 7,809 \| Tue: 7,005 \| Wed: 6,556<br>• Thu: 6,248 \| Fri: 6,723 \| Sat: 5,848<br>• Sun: 204 | Day of the week significantly influences sales, with Mondays acting as the peak traffic driver and Sundays virtually flat due to closures. |
| **State Holiday Effect** | **Sales collapse during holidays:**<br>• No Holiday: ~5,800–6,000<br>• Public Holiday: 291 \| Easter: 214 \| Christmas: 169 | The vast majority of physical storefronts remain closed during major state holidays. |
| **School Holiday Effect** | **Moderate positive impact:**<br>• No School Holiday: 5,621<br>• School Holiday: 6,477 | Households and families likely display higher shopping frequencies and spend volumes during school breaks. |
| **Promo Effect** | **Strong positive impact:**<br>• No Promo: 4,406<br>• Active Promo: 7,991 | Running promotional campaigns increases average sales by **~81%**, making it a primary performance driver. |
| **Store Type Effect** | **Store Type B dominates revenue:**<br>• Type B: 10,059 \| Type A: 5,738<br>• Type C: 5,724 \| Type D: 5,642 | Store Type B locations generate significantly higher revenue per store compared to all other formats. |
| **Assortment Effect** | **Larger assortments improve sales:**<br>• Extra: 8,554 \| Extended: 6,059 \| Basic: 5,481 | Offering a wider variety of product lines successfully attracts a larger customer volume and higher basket sizes. |
| **Competition Distance** | **Close competitors correlate with higher sales:**<br>• Very Close: 6,096 \| Close: 5,758<br>• Far: 5,680 \| Moderate: 5,570 | High-traffic commercial locations naturally attract both Rossmann storefronts and competing brands simultaneously. |
| **Promo2 Effect** | **Promo2 stores show lower average sales:**<br>• Promo2 = 0: 6,125<br>• Promo2 = 1: 5,424 | This counterintuitive drop likely reflects a selection bias, where continuous promotions are deployed specifically to stabilize underperforming stores. |

---

## Core Project Takeaways
1. **Feature Engineering Targets:** Day of the week, promotional status (`Promo`), and store closures (`Open`) represent the highest-variance features for baseline predictive performance.
2. **Data Cleaning:** Rows where `Open = 0` should be handled carefully during training since they introduce deterministic zeroes into the target variable (`Sales`).

# Phase 1 Findings: Classical Time Series Forecasting

## Objective

The objective of Phase 1 was to establish a forecasting benchmark by evaluating classical time series models on aggregated daily Rossmann sales data.

Since the original dataset contains sales information for 1,115 stores, store-level sales were aggregated into a single daily sales series:

**Date → Total Daily Sales**

This transformed the problem into a univariate time series forecasting task.

---

# Models Evaluated

The following forecasting approaches were implemented:

| Category | Models |
|---|---|
| Baseline Models | Naive Forecast, Seasonal Naive Forecast |
| Smoothing Models | Moving Average, Weighted Moving Average |
| Exponential Smoothing Models | Simple Exponential Smoothing (SES), Holt's Trend Model, Holt-Winters Triple Exponential Smoothing |

---

# Model Performance Analysis

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Holt-Winters | 1,250,170 | 1,789,503 | 88.84% |
| SES | 2,333,334 | 3,228,629 | 407.20% |
| Moving Average | 2,306,305 | 3,264,030 | 438.61% |
| Weighted Moving Average | 2,408,865 | 3,476,268 | 489.63% |
| Naive Forecast | 2,687,301 | 3,851,607 | 543.02% |
| Seasonal Naive | 2,911,575 | 4,072,885 | 119.21% |
| Holt | 5,190,674 | 6,321,653 | 712.32% |

---

# Analytical Insights

## 1. Trend Analysis

The aggregated sales series showed a gradual upward movement from 2013 to 2015.

This indicates that the business experienced overall sales growth over time.

Possible business factors contributing to this trend:

- Increasing customer base
- Store performance improvement
- Promotional activities
- Market expansion

Models incorporating trend information performed better than simple baseline approaches.

---

## 2. Seasonality Analysis

Exploratory analysis identified strong weekly seasonality.

Average sales varied significantly by day of the week:

| Day | Average Sales |
|---|---:|
| Monday | 7809 |
| Tuesday | 7005 |
| Wednesday | 6555 |
| Thursday | 6247 |
| Friday | 6723 |
| Saturday | 5847 |
| Sunday | 204 |

The large difference between weekdays and Sundays indicates strong recurring customer behaviour patterns.

This explains why:

- Holt-Winters performed best
- Holt's trend-only model performed poorly

because Holt-Winters models both trend and seasonality.

---

## 3. Model Behaviour Analysis

### Holt-Winters (Best Performing Model)

Holt-Winters achieved the lowest RMSE:

**RMSE: 1.79 million**

The model successfully captured:

- Level component → average sales behaviour
- Trend component → long-term growth
- Seasonal component → weekly sales cycles

Therefore, it was the most suitable classical forecasting approach for this dataset.

---

### Simple Exponential Smoothing (SES)

SES performed better than most baseline models because it adapts to recent sales levels.

However, it cannot capture:

- Growth trends
- Weekly seasonal behaviour

---

### Holt's Trend Model

Holt's model captures:

- Level
- Trend

but ignores seasonality.

Since Rossmann sales contain strong weekly patterns, ignoring seasonality resulted in the highest RMSE.

---

# Error Metric Interpretation

RMSE and MAE were considered the primary evaluation metrics.

MAPE values were relatively high because Rossmann contains a large number of zero-sales records caused by store closures.

When actual sales values are close to zero, percentage-based errors become unstable.

Therefore:

Primary metrics:
- RMSE
- MAE

Secondary metric:
- MAPE

---

# Business Interpretation

The analysis shows that Rossmann sales forecasting cannot rely only on historical averages.

Sales are influenced by:

- Time-based trends
- Weekly customer behaviour
- Seasonal patterns

A model that understands these patterns provides more accurate demand forecasts.

Accurate sales forecasting can help businesses:

- Optimize inventory planning
- Improve staff allocation
- Plan promotional campaigns
- Reduce stock shortages and excess inventory

---

# Phase 1 Conclusion

Holt-Winters Triple Exponential Smoothing was selected as the best classical forecasting model.

It provides the benchmark performance for the next phase, where advanced statistical models such as ARIMA, SARIMA, and SARIMAX will be evaluated.

| Test              | Finding                        | Interpretation                          |
| ----------------- | ------------------------------ | --------------------------------------- |
| ADF Test          | p-value = 0.000064             | Daily sales series is stationary        |
| ACF               | Strong spikes at 7, 14, 21, 28 | Weekly seasonality exists               |
| PACF              | Strong spikes at 7 and 14      | Weekly autoregressive behavior          |
| Seasonal Period   | 7 days                         | Retail sales repeat weekly              |
| Modeling Decision | SARIMA required                | ARIMA alone may miss seasonal structure |

| Test                     | Result                    | Conclusion                   |
| ------------------------ | ------------------------- | ---------------------------- |
| ADF (Original Series)    | p = 0.000064              | Stationary                   |
| ADF (Differenced Series) | p = 3.75e-26              | Highly stationary            |
| Differencing Required?   | No                        | d = 0 selected               |
| Seasonal Pattern         | Strong at 7,14,21,28 lags | Weekly seasonality confirmed |

| Model            | Observation                                 |
| ---------------- | ------------------------------------------- |
| ARIMA(1,0,1)     | Failed to capture weekly sales cycles       |
| Error Metrics    | Higher than Holt-Winters                    |
| Business Insight | Weekly seasonality dominates sales behavior |
| Next Step        | Introduce seasonal component using SARIMA   |

SARIMAX
│
├── ARIMA
├── Seasonal ARIMA (SARIMA)
├── ARIMAX
└── SARIMAX

| Metric | Winner                        |
| ------ | ----------------------------- |
| MAE    | Holt-Winters (1.25M vs 1.30M) |
| RMSE   | Holt-Winters (1.79M vs 1.91M) |
| MAPE   | SARIMA (71.61% vs 88.84%)     |

SARIMA dramatically outperformed ARIMA,
confirming that weekly seasonality is a major
driver of Rossmann sales.

The seasonal AR coefficient (0.9987)
indicates strong dependence on sales
from the same weekday in previous weeks.

Compared with Holt-Winters,
SARIMA achieved lower MAPE but slightly
higher MAE and RMSE.

| Analysis                   | Finding                                                                                  |
| -------------------------- | ---------------------------------------------------------------------------------------- |
| SARIMA(7,0,0)(1,0,1,7)     | Model could not be fitted due to overlapping lag 7 in seasonal and non-seasonal AR terms |
| Root Cause                 | Lag 7 was included in both AR(7) and Seasonal AR(1, period=7) components                 |
| Statistical Interpretation | Weekly seasonality is already explicitly modeled through the seasonal component          |
| Modeling Decision          | Replace AR(7) with a lower-order AR component such as AR(5)                              |
| Next Step                  | Evaluate SARIMA(5,0,0)(1,0,1,7) and compare with existing SARIMA model                   |

| Analysis            | Finding                                                                                                           |
| ------------------- | ----------------------------------------------------------------------------------------------------------------- |
| ACF Analysis        | Strong seasonal spikes observed at lags 7, 14, 21, and 28                                                         |
| PACF Analysis       | Weekly autoregressive behavior detected                                                                           |
| Modeling Constraint | SARIMA(7,0,0)(1,0,1,7) is invalid because lag 7 appears in both seasonal and non-seasonal AR terms                |
| Revised Model       | SARIMA(5,0,0)(1,0,1,7) selected to capture short-term autoregressive effects while preserving weekly seasonality  |
| Objective           | Determine whether a higher-order AR component improves forecasting performance compared to SARIMA(1,0,1)(1,0,1,7) |

| Analysis                | Finding                                                                                                           |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Stationarity Test (ADF) | Daily sales series is stationary (p-value < 0.05), therefore differencing was not required                        |
| ACF Analysis            | Strong seasonal spikes observed at lags 7, 14, 21, and 28, indicating weekly seasonality                          |
| PACF Analysis           | Significant spikes at lag 7 and lag 14 suggested autoregressive seasonal behavior                                 |
| ARIMA(1,0,1)            | Failed to capture weekly seasonality, resulting in high forecasting error                                         |
| SARIMA(1,0,1)(1,0,1,7)  | Successfully modeled weekly sales cycles and substantially reduced forecasting error                              |
| Seasonal AR Coefficient | Seasonal AR(7) coefficient ≈ 0.999, indicating strong dependence on sales from the same weekday in previous weeks |
| SARIMA(5,0,0)(1,0,1,7)  | Additional AR terms increased model complexity without improving forecast accuracy                                |
| Phase 2 Winner          | SARIMA(1,0,1)(1,0,1,7) achieved the best balance of accuracy and model simplicity                                 |
| Business Insight        | Weekly purchasing behavior is the dominant driver of Rossmann sales patterns                                      |
| Modeling Insight        | Incorporating seasonality is more important than increasing non-seasonal autoregressive order                     |

| Analysis                | Finding                                                                                                    |
| ----------------------- | ---------------------------------------------------------------------------------------------------------- |
| Prophet Performance     | Prophet achieved MAE = 1.51M, RMSE = 2.10M, and MAPE = 107.32%                                             |
| Trend Modeling          | Prophet successfully captured the long-term sales trend                                                    |
| Seasonality Modeling    | Weekly and yearly seasonal components were automatically learned                                           |
| Comparative Performance | Prophet underperformed both Holt-Winters and SARIMA models                                                 |
| Best Statistical Model  | SARIMA(1,0,1)(1,0,1,7) remained the strongest forecasting model                                            |
| Key Insight             | Explicit modeling of weekly seasonality produced more accurate forecasts than Prophet's automated approach |
| Business Conclusion     | Rossmann sales are primarily driven by stable recurring weekly patterns rather than complex trend shifts   |
| Phase 3 Conclusion      | Prophet provides interpretable forecasts but was not the most accurate model for this dataset              |

## phase 4
| Analysis             | Finding                                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Dataset Integration  | Training and store metadata were merged using Store ID                                                              |
| Final Dataset        | Combined dataset contains daily sales transactions along with store-level business attributes                       |
| Data Size            | Dataset contains 1,017,209 sales records across 1,115 stores                                                        |
| Feature Availability | ML model will leverage promotional, competitive, holiday, and store characteristics                                 |
| Missing Values       | Missing values are mainly related to optional business factors such as competitor information and promotion history |

| Analysis               | Finding                                                                                               |
| ---------------------- | ----------------------------------------------------------------------------------------------------- |
| Date Transformation    | Date column was converted into Year, Month, Day, and WeekOfYear features to capture temporal patterns |
| Missing Value Handling | Missing competitor and promotion information was treated as absence of business activity              |
| Competition Features   | CompetitionDistance missing values were replaced using median imputation                              |
| Promotion Features     | Missing Promo2 details were replaced with zero/default categories                                     |
| Feature Expansion      | Original 18 columns were transformed into ML-ready features capturing temporal and business behavior  |

| Analysis            | Finding                                                                                                            |
| ------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Feature Engineering | Additional business features were created to improve model understanding of retail behavior                        |
| Competition Feature | CompetitionAge captures the impact of competitor presence over time                                                |
| Promotion Feature   | Promo2Age represents the duration of long-term promotional campaigns                                               |
| Temporal Feature    | Weekend indicators were added to capture weekly customer patterns                                                  |
| Data Filtering      | Closed-store records were removed because they contain zero sales and do not represent demand prediction scenarios |

| Analysis            | Finding                                                                                           |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| Feature Engineering | Created additional business features to improve ML model understanding of retail patterns         |
| CompetitionAge      | Captures duration of competitor presence near each store                                          |
| Promo2Age           | Represents duration of long-term promotional campaigns                                            |
| Weekend Behavior    | Added weekend indicator to capture weekly customer variations                                     |
| Data Cleaning       | Removed closed-store records since they contain zero sales and do not represent demand prediction |
| Feature Refinement  | Removed constant Open indicators after filtering inactive stores                                  |

| Analysis          | Finding                                                                                            |
| ----------------- | -------------------------------------------------------------------------------------------------- |
| Dataset Filtering | Removed closed-store records, reducing dataset from 1,017,209 to 844,392 active sales observations |
| CompetitionAge    | Created feature representing competitor presence duration; average competition age was 3.56 years  |
| Promo2Age         | Created promotion duration feature with average duration of 1.08 years                             |
| Weekend Behavior  | Added weekend indicator; approximately 17.5% of observations occurred on weekends                  |
| Feature Expansion | Original business attributes were transformed into 24 ML-ready features                            |

| Analysis           | Finding                                                                                                                 |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Data Type Cleaning | Converted categorical variables into consistent string format before encoding to handle mixed numerical and text values |
| Encoding Strategy  | Applied Label Encoding to transform categorical business attributes into numerical representations                      |

| Analysis             | Finding                                                                                  |
| -------------------- | ---------------------------------------------------------------------------------------- |
| Categorical Encoding | Converted categorical variables into numerical representations for ML compatibility      |
| Encoded Variables    | StateHoliday, StoreType, Assortment, and PromoInterval were label encoded                |
| Data Type Handling   | Converted mixed string/integer categories into consistent formats before encoding        |
| Feature Cleanup      | Removed constant IsOpen feature since all remaining observations represent active stores |

| Analysis                | Finding                                                                           |
| ----------------------- | --------------------------------------------------------------------------------- |
| Data Splitting Strategy | Used chronological train-test split to preserve real-world forecasting conditions |
| Avoided Random Split    | Prevented future information leakage into model training                          |
| Training Data           | Earlier historical sales records were used for model learning                     |
| Testing Data            | Future observations were reserved for unbiased evaluation                         |

### Baseline Model

A mean-based regression baseline was implemented to establish a reference performance level before applying machine learning algorithms.

The model predicts the average historical sales value for every test observation.

Evaluation metrics:

| Metric | Result |
|---|---:|
| MAE | 2239.94 |
| RMSE | 3070.02 |

MAPE was not considered reliable because zero-sales observations cause division instability. Future model evaluation will use a zero-safe MAPE calculation.

This baseline provides a minimum benchmark that advanced ML models must outperform.

### Models Implemented

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Baseline Mean Predictor | 2239.94 | 3070.02 | - |
| Linear Regression | 1991.76 | 2761.12 | 42.31 |
| Random Forest | 804.17 | 1190.38 | 11.43 |

Top influencing features(under RF):

| Feature | Importance | Business Interpretation |
|---|---:|---|
| CompetitionDistance | 20.09% | Store location relative to competitors strongly affects sales |
| Store | 16.61% | Individual store characteristics significantly influence demand |
| Promo | 14.83% | Promotional campaigns have a major impact on sales |
| Competition Opening Features | ~13% | Competitor presence affects store performance |
| DayOfWeek | 6.47% | Weekly customer behaviour influences demand |
| WeekOfYear | 4.00% | Seasonal patterns contribute to sales variation |
### Key Insights

- Sales prediction is strongly influenced by competitive environment and store-level characteristics.
- Promotional activities are a major controllable factor affecting revenue.
- Calendar-based features capture recurring demand patterns.
- The importance of Store ID indicates significant variation between locations, suggesting store-specific modelling can further improve predictions.
# 4.7 XGBoost Regression

XGBoost (Extreme Gradient Boosting) was implemented as an advanced tree-based ensemble model to further improve prediction accuracy.

Unlike Random Forest, which builds independent decision trees, XGBoost builds trees sequentially where each new tree corrects the errors made by previous trees.

## Performance
| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Mean Baseline | 2239.94 | 3070.02 | - |
| Linear Regression | 1991.76 | 2761.12 | 42.31% |
| Random Forest | 804.17 | 1190.38 | 11.43% |
| XGBoost | **792.78** | **1137.41** | 11.59% |

## Findings

- XGBoost achieved the lowest MAE and RMSE among all evaluated machine learning models.
- Compared with Random Forest:
  - MAE improved from 804.17 → 792.78
  - RMSE improved from 1190.38 → 1137.41
- The improvement indicates that gradient boosting captured additional non-linear relationships between:
  - Store characteristics
  - Competition factors
  - Promotional campaigns
  - Calendar-based demand patterns

Although MAPE is marginally higher than Random Forest, the overall error reduction makes XGBoost the strongest performing model so far.
# 4.9 Model Saving and Deployment Preparation

The final XGBoost model was saved using Joblib for future predictions.

Saved model:models/xgboost_sales_forecaster.pkl

The saved model can be loaded independently without retraining, enabling future integration with:

- Streamlit dashboard
- Prediction API
- Automated forecasting pipeline

## Final Phase 4 Outcome

A complete machine learning forecasting pipeline was developed:
# 4.10 Model Validation After Saving

The saved XGBoost model was loaded independently and tested on unseen test samples to ensure that the serialization process preserved model performance.

Example predictions:

| Actual Sales | Predicted Sales |
|---:|---:|
| 5942 | 6579.84 |
| 8423 | 8609.78 |
| 7642 | 7046.03 |
| 6338 | 6696.49 |
| 9972 | 8763.07 |

## Validation Findings

- The loaded model successfully generated predictions without retraining.
- Predicted values closely followed actual sales values.
- This confirms that the trained model can be reused for future inference.

The final machine learning pipeline is now ready for integration into an application layer.

# Phase 1 Findings: Classical Time Series Forecasting

| Analysis | Finding |
|---|---|
| Dataset Preparation | Retail sales data was aggregated into daily time-series format for forecasting |
| Forecasting Objective | The objective was to predict future sales patterns using historical demand behaviour |
| Time-Series Structure | Sales data was ordered chronologically to preserve temporal dependencies |
| Data Frequency | Daily sales observations were used to capture short-term and weekly demand variations |
| Dataset Split | Historical observations were divided into training and testing periods for unbiased evaluation |

| Analysis | Finding |
|---|---|
| Exploratory Analysis | Sales trends, fluctuations, and recurring patterns were analysed before modelling |
| Trend Behaviour | Sales showed variation over time indicating changing demand patterns |
| Seasonal Behaviour | Weekly recurring patterns were observed, suggesting strong day-based seasonality |
| Forecasting Challenge | Retail sales contain irregular demand fluctuations caused by promotions, holidays, and business factors |
| Model Requirement | Multiple forecasting approaches were tested to identify the most suitable baseline model |

| Analysis | Finding |
|---|---|
| Stationarity Testing | Augmented Dickey-Fuller (ADF) test was performed to check time-series stationarity |
| ADF Result | P-value = 6.43 × 10⁻⁵ indicating the sales series is stationary |
| Differencing Requirement | Additional differencing was not required for ARIMA-based models |
| Statistical Behaviour | The series maintained stable mean and variance characteristics suitable for forecasting |

| Analysis | Finding |
|---|---|
| Autocorrelation Analysis | ACF and PACF plots were analysed to identify temporal relationships |
| Weekly Seasonality | Strong autocorrelation peaks were observed at lag 7, 14, 21, and 28 |
| Seasonal Pattern | The presence of 7-day repeating behaviour confirmed weekly seasonality |
| Model Selection Impact | Seasonal forecasting models were considered due to observed weekly demand cycles |

| Analysis | Finding |
|---|---|
| Naive Forecast | Used previous sales value as future prediction baseline |
| Seasonal Naive | Used previous week's sales pattern to capture weekly repetition |
| Moving Average | Smoothed short-term fluctuations using historical averages |
| Weighted Moving Average | Assigned higher importance to recent observations |
| SES | Captured level-based forecasting behaviour |
| Holt Method | Modelled trend-based patterns |
| Holt-Winters | Incorporated level, trend, and seasonality components |

# Model Performance Comparison

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Naive Forecast | 2687301 | 3851607 | 543.02% |
| Seasonal Naive | 2911575 | 4072885 | 119.21% |
| Moving Average | 2306305 | 3264030 | 438.61% |
| Weighted Moving Average | 2408865 | 3476268 | 489.63% |
| SES | 2333334 | 3228629 | 407.20% |
| Holt | 5190674 | 6321653 | 712.32% |
| Holt-Winters | **1250170** | **1789503** | **88.84%** |

# Best Classical Forecasting Model

## Holt-Winters Exponential Smoothing

Holt-Winters achieved the best forecasting performance among classical statistical models.

## Findings

- Holt-Winters achieved the lowest MAE and RMSE.
- The model effectively captured:
  - Weekly seasonal behaviour
  - Underlying sales level
  - Short-term demand fluctuations
- Seasonal models significantly outperformed simple forecasting approaches.
- The results confirm that retail sales contain strong repeating weekly patterns.

# Business Insights

- Retail demand is strongly influenced by recurring weekly customer behaviour.
- Simple forecasting methods struggle with irregular retail fluctuations.
- Incorporating seasonality improves forecast accuracy significantly.
- Classical forecasting models provide a strong benchmark before applying advanced machine learning approaches.

# Phase 1 Outcome

A complete classical time-series forecasting pipeline was developed:


# Phase 2 Findings: ARIMA & SARIMA Forecasting

| Analysis | Finding |
|---|---|
| Objective | Advanced statistical forecasting models were implemented to improve upon classical forecasting benchmarks |
| Approach | ARIMA and SARIMA models were evaluated using autoregressive, moving average, and seasonal components |
| Model Selection | ACF and PACF analysis from Phase 1 guided AR and MA parameter selection |
| Seasonal Pattern | Weekly seasonality identified in Phase 1 was incorporated using seasonal SARIMA components |

| Analysis | Finding |
|---|---|
| Stationarity Testing | Augmented Dickey-Fuller (ADF) test was performed before ARIMA modelling |
| ADF Statistic | -4.7616 |
| P-value | 6.43 × 10⁻⁵ |
| Interpretation | The sales series was stationary, therefore additional differencing was not required |
| Differencing Order | Integrated component was set as d = 0 |

| Analysis | Finding |
|---|---|
| ACF Analysis | Strong autocorrelation peaks were observed at seasonal lags |
| Significant Lags | Lag 7, 14, 21, and 28 showed strong positive autocorrelation |
| Seasonal Period | Weekly seasonality with period 7 was confirmed |
| Model Decision | Seasonal ARIMA models were preferred over non-seasonal ARIMA |

| Analysis | Finding |
|---|---|
| ARIMA Model | ARIMA(1,0,1) was implemented as a non-seasonal benchmark |
| SARIMA Model | SARIMA models incorporated weekly seasonal behaviour |
| Parameter Selection | Models were iteratively tested based on statistical significance and evaluation metrics |
| Evaluation Criteria | Models were compared using MAE, RMSE, and MAPE |

# Model Performance Comparison

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| ARIMA(1,0,1) | 2520905.64 | 3264490.02 | 381.57% |
| SARIMA(1,0,1)x(1,0,1,7) | **1302160.69** | **1913409.13** | **71.61%** |
| SARIMA(5,0,0)x(1,0,[1],7) | 1507592.87 | 2128379.78 | 86.62% |

# Best Statistical Forecasting Model

## SARIMA(1,0,1)x(1,0,1,7)

SARIMA provided the strongest performance among ARIMA-family models.

## Findings

- Seasonal SARIMA significantly improved forecasting accuracy compared with standard ARIMA.
- Incorporating weekly seasonality reduced forecasting errors.
- The seasonal autoregressive component captured recurring weekly demand behaviour.
- SARIMA achieved lower MAE and RMSE compared with non-seasonal ARIMA.

# Statistical Model Diagnostics

| Analysis | Finding |
|---|---|
| Ljung-Box Test | Residuals showed no significant remaining autocorrelation |
| Seasonal Component | Weekly seasonal dependency was successfully captured |
| Model Limitation | Residual distribution remained non-normal due to retail demand volatility |
| Business Impact | Statistical models provide reliable short-term forecasts but cannot fully capture external business factors |

# Business Insights

- Retail sales exhibit strong weekly recurring behaviour.
- Seasonal modelling is essential for accurate retail forecasting.
- ARIMA models alone are limited because they only use historical sales patterns.
- External factors such as promotions, competition, and store characteristics require machine learning approaches.

# Phase 2 Outcome

A complete ARIMA-family forecasting pipeline was developed:


# Phase 3 Findings: Prophet Time Series Forecasting

| Analysis | Finding |
|---|---|
| Objective | Prophet forecasting model was implemented as an alternative time-series approach |
| Model Type | Prophet is a decomposable forecasting model based on trend and seasonality components |
| Forecasting Components | Model captures trend, weekly seasonality, yearly seasonality, and uncertainty intervals |
| Purpose | Evaluate whether automated seasonal decomposition improves forecasting performance |

| Analysis | Finding |
|---|---|
| Data Preparation | Sales data was converted into Prophet-required format with date column (ds) and target variable (y) |
| Training Strategy | Historical sales data was used for training while future observations were reserved for evaluation |
| Forecast Horizon | Future sales values were generated for the testing period |
| Output Generation | Prophet produced predicted sales values along with confidence intervals |

| Analysis | Finding |
|---|---|
| Trend Component | Prophet captured the underlying long-term sales movement |
| Weekly Seasonality | Weekly demand variations were incorporated automatically |
| Yearly Seasonality | Annual patterns were considered during forecasting |
| Forecast Flexibility | Model handled changing trends without manual parameter tuning |

# Prophet Model Performance

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Prophet | 1510428.54 | 2101397.93 | 107.32% |

# Findings

- Prophet successfully captured overall sales trends and seasonal behaviour.
- Forecast performance was competitive with classical statistical models.
- The model automatically detected seasonal patterns without requiring manual ACF/PACF analysis.
- Higher MAPE indicates difficulty in accurately predicting low-sales observations.

# Comparison With Previous Forecasting Approaches

| Model Category | Strength | Limitation |
|---|---|---|
| Classical Models | Strong performance on historical patterns and seasonality | Limited business feature understanding |
| ARIMA/SARIMA | Captures autocorrelation and seasonal dependencies | Requires manual parameter tuning |
| Prophet | Automatic trend and seasonality modelling | Less effective when external business variables dominate |

# Business Insights

- Sales forecasting requires both temporal understanding and business context.
- Trend and seasonality alone cannot fully explain retail demand fluctuations.
- Promotional campaigns, store characteristics, and competition effects must be incorporated for improved prediction accuracy.

# Phase 3 Outcome

A complete Prophet forecasting pipeline was developed:
