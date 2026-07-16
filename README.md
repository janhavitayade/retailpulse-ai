# 🛍️ RetailPulse AI
### Forecasting the Pulse of Rossmann's Retail Sales

> An end-to-end sales forecasting journey — from simple averages to gradient boosting — built on real-world retail data from **1,115 Rossmann drugstores** across Germany.

🔗 **Live Demo:** [retailpulse-forecasting-ai.streamlit.app](https://retailpulse-forecasting-ai.streamlit.app/)
📄 **Full Project Report (PDF):** [RetailPulse_AI_Summary_Report.pdf](./reports/RetailPulse_AI_Summary_Report.pdf)
📦 **Data Source:** [Kaggle — Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales/data)

---

## 📖 The Story

Every day, across 1,115 Rossmann stores, cash registers ring up sales shaped by forces invisible to a spreadsheet — a Monday rush, a promotion, a nearby competitor opening its doors, a public holiday shutting everything down. **RetailPulse AI** is the story of turning that noise into a signal: a journey through four modeling phases, each one asking the same question a little more precisely — *"What will tomorrow's sales look like?"*

This project doesn't just fit a model and stop. It's a progression — starting with the simplest possible forecasts, layering in statistical rigor, testing automated decomposition, and finally arriving at a machine learning model that understands *why* sales move the way they do.

---

## 📦 The Dataset

The foundation of this story is Rossmann's historical daily sales data — **1,017,209 records across 1,115 stores**, spanning **January 2013 to July 2015** (942 unique days).

| Field | Description |
|---|---|
| **Id** | An Id representing a (Store, Date) pair, used only in the test set |
| **Store** | A unique Id for each store |
| **Sales** | The turnover for a given day *(the target variable)* |
| **Customers** | Number of customers on a given day |
| **Open** | Whether the store was open — `0` = closed, `1` = open |
| **StateHoliday** | `a` = public holiday, `b` = Easter, `c` = Christmas, `0` = none. Nearly all stores close on state holidays; schools close on public holidays and weekends |
| **SchoolHoliday** | Whether the (Store, Date) was affected by public school closures |
| **StoreType** | Differentiates 4 store models: `a`, `b`, `c`, `d` |
| **Assortment** | Assortment level: `a` = basic, `b` = extra, `c` = extended |
| **CompetitionDistance** | Distance (in meters) to the nearest competitor store |
| **CompetitionOpenSince[Month/Year]** | Approximate month/year the nearest competitor opened |
| **Promo** | Whether the store is running a promo that day |
| **Promo2** | Continuing, consecutive promotion — `0` = not participating, `1` = participating |
| **Promo2Since[Year/Week]** | Year and calendar week the store joined Promo2 |
| **PromoInterval** | Months when a new Promo2 round starts, e.g. `"Feb,May,Aug,Nov"` |

**Data quality snapshot:**

| Check | Finding |
|---|---|
| Missing values (train) | None |
| Missing values (store) | Present in competition & Promo2 fields — business-driven, not errors |
| Zero-sales records | 172,871 total → **172,817** occurred while `Open = 0`; only 54 occurred while open |
| Sales distribution | Mean: 5,773.82 · Median: 5,744 · Std Dev: 3,849.93 · Max: 41,551 |

The takeaway from the very first look at the data: **closed stores, not weak demand, explain almost every zero-sales day.** That single insight would go on to shape every phase that followed.

---

## 🧭 Chapter 1 — Phase 1: Learning to Walk (Classical Forecasting)

*App page: `📊 Phase 1 - Data Understanding`*

Before reaching for anything sophisticated, the story starts with the basics: aggregate every store's sales into one daily total, and see what patterns emerge.

Two things jumped out immediately:

- A **gradual upward trend** in sales from 2013 to 2015.
- A **strong weekly rhythm** — Mondays roaring, Sundays going nearly silent.

| Day | Average Sales |
|---|---:|
| Monday | 7,809 |
| Tuesday | 7,005 |
| Wednesday | 6,556 |
| Thursday | 6,248 |
| Friday | 6,723 |
| Saturday | 5,848 |
| Sunday | 204 |

Seven classical models were put to the test — from a naive "tomorrow = today" guess up to full Holt-Winters smoothing:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| **Holt-Winters** | **1,250,170** | **1,789,503** | **88.84%** |
| SES | 2,333,334 | 3,228,629 | 407.20% |
| Moving Average | 2,306,305 | 3,264,030 | 438.61% |
| Weighted Moving Average | 2,408,865 | 3,476,268 | 489.63% |
| Naive Forecast | 2,687,301 | 3,851,607 | 543.02% |
| Seasonal Naive | 2,911,575 | 4,072,885 | 119.21% |
| Holt | 5,190,674 | 6,321,653 | 712.32% |

**Why Holt-Winters won:** it was the only model to explicitly capture *all three* ingredients present in the data — level, trend, and weekly seasonality. Holt's trend-only model, by contrast, ignored seasonality entirely and paid for it with the worst error of the lot.

> 🧠 *MAPE values look high across the board — that's because store closures create a flood of near-zero sales days, and percentage errors become unstable near zero. MAE and RMSE were treated as the primary metrics; MAPE as secondary.*

**Phase 1 verdict:** Holt-Winters sets the benchmark. Sales clearly aren't random — they trend, and they repeat weekly. Phase 2 asks whether a more statistically rigorous approach can do even better.

---

## 🧭 Chapter 2 — Phase 2: Getting Statistical (ARIMA & SARIMA)

*App page: `📈 Phase 2 - ARIMA & SARIMA`*

With a benchmark in hand, the story turns to formal time-series statistics — starting with a simple question: **is this series even stationary?**

| Test | Finding | Interpretation |
|---|---|---|
| ADF (original series) | p = 6.43 × 10⁻⁵ | Series is stationary |
| ADF (differenced series) | p = 3.75 × 10⁻²⁶ | Highly stationary |
| Differencing needed? | No | `d = 0` selected |

ACF and PACF plots told a consistent story: sharp spikes at lags **7, 14, 21, and 28** — an unmistakable weekly fingerprint. Plain ARIMA, which knows nothing about seasonality, was always going to struggle.

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| ARIMA(1,0,1) | 2,520,905.64 | 3,264,490.02 | 381.57% |
| **SARIMA(1,0,1)(1,0,1,7)** | **1,302,160.69** | **1,913,409.13** | **71.61%** |
| SARIMA(5,0,0)(1,0,[1],7) | 1,507,592.87 | 2,128,379.78 | 86.62% |

A quick detour: an attempt at SARIMA(7,0,0)(1,0,1,7) failed outright — lag 7 was double-booked in both the seasonal and non-seasonal AR terms. Swapping in a lower-order AR(5) fixed the conflict, but it didn't beat the simpler model.

**The winner — SARIMA(1,0,1)(1,0,1,7)** — carried a seasonal AR coefficient of **≈0.999**, meaning: *whatever happened on this weekday last week is an extremely strong predictor of today.* Adding more autoregressive complexity (AR(5)) didn't help; the weekly seasonal term was already doing the heavy lifting.

| Metric | Better Model |
|---|---|
| MAE | Holt-Winters (1.25M vs 1.30M) |
| RMSE | Holt-Winters (1.79M vs 1.91M) |
| MAPE | **SARIMA (71.61% vs 88.84%)** |

**Phase 2 verdict:** SARIMA formalizes what Holt-Winters found intuitively — weekly seasonality is the dominant force in this data. Residual diagnostics (Ljung-Box) confirmed the seasonal structure was fully captured, though real-world volatility kept residuals non-normal. Statistical models, however, only ever look backward at sales history — they know nothing about *why* those patterns exist. Phase 3 tries a more automated lens.

---

## 🧭 Chapter 3 — Prophet: The Automated Storyteller

*App page: `🔮 Phase 3 - Prophet Forecasting`*

Facebook's Prophet promised something appealing: automatic decomposition of trend, weekly seasonality, and yearly seasonality — no manual ACF/PACF detective work required.

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Prophet | 1,510,428.54 | 2,101,397.93 | 107.32% |

Prophet did successfully identify the long-term upward trend and both weekly and yearly seasonal cycles — all without any manual tuning. But when placed next to the competition, it came in behind both:

| Model Family | Strength | Limitation |
|---|---|---|
| Classical (Holt-Winters) | Strong on historical seasonality | No business-context awareness |
| ARIMA/SARIMA | Captures autocorrelation precisely | Requires manual parameter tuning |
| **Prophet** | Fully automated decomposition | Weaker when external business factors dominate |

**Phase 3 verdict:** convenience isn't the same as accuracy. Prophet's hands-off approach couldn't out-perform SARIMA's deliberately-tuned seasonal modeling. More importantly, it exposed the real limitation of *every* model so far: none of them knew anything about promotions, competitors, or store types. That's where the story finally turns to machine learning.

---

## 🧭 Chapter 4 — Machine Learning: Teaching the Model *Why*

*App page: `🤖 Phase 4 - Machine Learning`*

This chapter changes the question. Instead of "what happened historically on this date?", it asks: **"what business conditions were in play?"** Sales data was merged with store metadata — competition distance, store type, assortment, promotions — and engineered into **24 ML-ready features**, including `CompetitionAge`, `Promo2Age`, and weekend indicators. Closed-store records (zero-sales-by-definition) were removed, trimming the dataset from 1,017,209 to **844,392 active observations**.

A chronological train-test split preserved real-world forecasting conditions — no future data leaking backward into training.

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Mean Baseline | 2,239.94 | 3,070.02 | — |
| Linear Regression | 1,991.76 | 2,761.12 | 42.31% |
| Random Forest | 804.17 | 1,190.38 | 11.43% |
| **XGBoost** | **792.78** | **1,137.41** | 11.59% |

*(Note the scale shift: these are per-store daily errors, not aggregated totals — this is why the numbers look so much smaller than earlier phases.)*

**What actually drives sales**, per Random Forest feature importance:

| Feature | Importance | Business Meaning |
|---|---:|---|
| CompetitionDistance | 20.09% | Location relative to competitors matters a lot |
| Store (identity) | 16.61% | Individual store characteristics vary significantly |
| Promo | 14.83% | Promotions strongly move sales |
| Competition opening features | ~13% | Timing of competitor entry affects performance |
| DayOfWeek | 6.47% | Weekly rhythm still matters |
| WeekOfYear | 4.00% | Seasonal calendar effects persist |

Other EDA-level findings that reinforced the feature importance results:

| Factor | Effect |
|---|---|
| Promo active | Sales rise ~81% (4,406 → 7,991 average) |
| Store Type B | Highest revenue (10,059) vs. A/C/D (~5,700–5,740) |
| Assortment "Extra" | Highest average sales (8,554) vs. Basic (5,481) |
| Promo2 active | *Lower* average sales (5,424 vs 6,125) — likely because Promo2 is deployed to prop up underperforming stores, not reward strong ones |

**XGBoost was selected as the final deployed model**, edging out Random Forest on both MAE and RMSE by capturing non-linear interactions between competition, promotions, and calendar effects that simpler models missed. The trained model was serialized with Joblib, reloaded independently, and validated against unseen samples to confirm predictions held up after saving — clearing it for deployment into this Streamlit app.

**Phase 4 verdict:** business context — not just historical sales patterns — is what ultimately separates a good forecast from a great one.

---

## 🏆 The Full Arc, at a Glance

| Phase | Approach | Best Model | Key Metric |
|---|---|---|---|
| 1 | Classical Smoothing | Holt-Winters | MAE 1.25M |
| 2 | Statistical (ARIMA family) | SARIMA(1,0,1)(1,0,1,7) | MAPE 71.61% |
| 3 | Automated Decomposition | Prophet | MAE 1.51M *(underperformed 1 & 2)* |
| 4 | Machine Learning | **XGBoost** | **MAE 792.78 (per-store)** |

---

## 🎯 The App Itself

The Streamlit app mirrors this journey exactly — each phase gets its own page, plus a live prediction tool:

- **🏠 Overview** — project summary, tech stack, modeling pipeline
- **📊 Phase 1** — Data understanding & classical forecasting visuals
- **📈 Phase 2** — ARIMA/SARIMA diagnostics and forecasts
- **🔮 Phase 3** — Prophet trend/seasonality decomposition
- **🤖 Phase 4** — Feature importance & ML model comparison
- **🎯 Sales Prediction** — Real-time sales forecast using the deployed XGBoost model

👉 **Try it live:** [retailpulse-forecasting-ai.streamlit.app](https://retailpulse-forecasting-ai.streamlit.app/)

---

## 🛠️ Technology Stack

`Python` · `Pandas` · `NumPy` · `Statsmodels` · `Prophet` · `Scikit-Learn` · `XGBoost` · `Streamlit`

---

## 📂 Project Structure

```text
RetailPulse-AI/
│
├── app/
│   └── app.py                             # Streamlit dashboard application
│
├── asset/
│   ├── icon.png                           # Application icon
│   │
│   ├── phase1/                            # Time Series Forecasting Assets
│   │   ├── sales_over_time.png
│   │   └── holt_winters_forecast.png
│   │
│   ├── phase2/                            # ARIMA & SARIMA Forecasting Assets
│   │   ├── acf_plot.png
│   │   ├── pacf_plot.png
│   │   ├── first_difference_series.png
│   │   ├── arima_forecast.png
│   │   └── sarima_forecast.png
│   │
│   ├── phase3/                            # Prophet Forecasting Assets
│   │   ├── prophet_forecast.png
│   │   ├── prophet_trend.png
│   │   ├── prophet_weekly.png
│   │   └── prophet_yearly.png
│   │
│   └── phase4/                            # Machine Learning Assets
│       ├── feature_importance.png
│       └── xgboost_actual_vs_predicted.png
│
├── data/
│   ├── raw/                               # Original Kaggle dataset files
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── store.csv
│   │   └── sample_submission.csv
│   │
│   └── processed/                         # Cleaned and transformed datasets
│
├── models/
│   ├── encoders.pkl                       # Saved categorical encoders
│   └── xgboost_sales_forecaster.pkl       # Trained XGBoost forecasting model
│
├── notebooks/
│   ├── 01_data_understanding.ipynb        # Dataset exploration and EDA
│   ├── 02_phase1_forecasting.ipynb        # Holt-Winters forecasting model
│   ├── 03_phase2_arima_sarima.ipynb       # ARIMA and SARIMA forecasting
│   ├── 04_phase3_prophet.ipynb            # Prophet forecasting model
│   └── 05_phase4_machine_learning.ipynb   # ML models and evaluation
│
├── reports/
│   └── RetailPulse_AI_Summary_Report.pdf  # Complete project report
│
├── requirements.txt                       # Project dependencies
├── README.md                              # Project documentation
└── .gitignore                             # Git ignored files

```

## ⚙️ Getting Started — Run This Project Yourself

Follow these commands to clone, set up, and run RetailPulse AI locally.

### 1. Clone the repository

```bash
git clone https://github.com/janhavitayade/retailpulse-ai.git
cd retailpulse-ai
```

### 2. Create and activate a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` isn't present yet, generate one after installing your packages with:
> ```bash
> pip freeze > requirements.txt
> ```
> At minimum, this project needs:
> ```bash
> pip install streamlit pandas numpy scikit-learn xgboost statsmodels prophet joblib matplotlib
> ```

### 4. Add the dataset

Download the data from Kaggle and place the CSVs inside `data/raw/`:

```
data/raw/
├── train.csv
├── test.csv
├── store.csv
└── sample_submission.csv
```

📦 Dataset: [Rossmann Store Sales — Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data)

*(You'll need a free Kaggle account and to accept the competition rules to download.)*

### 5. Run the notebooks (optional — to reproduce the model)

If you want to retrain the model from scratch instead of using the saved `.pkl` files:

```bash
jupyter notebook notebooks/01_data_understanding.ipynb
```

Run the notebooks in order:

```
01_data_understanding.ipynb        → EDA & data quality checks
02_phase1_forecasting.ipynb        → Holt-Winters & classical models
03_phase2_arima_sarima.ipynb       → ARIMA / SARIMA
04_phase3_prophet.ipynb            → Prophet forecasting
05_phase4_machine_learning.ipynb   → Random Forest & XGBoost (final model)
```

This will regenerate `models/xgboost_sales_forecaster.pkl` and `models/encoders.pkl`.

### 6. Launch the Streamlit app

```bash
streamlit run app/app.py
```

The app will open automatically at:

```
http://localhost:8501
```

### 7. (Optional) Deploy your own copy on Streamlit Cloud

1. Push your repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app** → select your repo → set the main file path to `app/app.py`.
4. Click **Deploy**.
