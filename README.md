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