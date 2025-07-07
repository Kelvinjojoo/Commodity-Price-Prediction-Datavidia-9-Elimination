# 🛒 Commodity Price Prediction - Datavidia 9.0 Preliminary Round (2025)

This repository contains my personal participation for the Datavidia 9.0 Preliminary Round, held in 2025. The task involves forecasting daily food commodity prices across 34 provinces in Indonesia, using various public datasets including local price data, Google Trends, currency rates, and global commodity prices.

On July 5th, 2025, I revisited and improved my approach to achieve better predictive accuracy and lower MAPE score.

The goal of this challenge is to build robust forecasting models and communicate data-driven insights that could potentially support decision-making in industries affected by price volatility - particularly in the food supply chain.


## 🎯 Background and Objective

Food price volatility is a major economic issue in Indonesia. It affects various sectors — from household purchasing power to the operational stability of culinary and catering businesses. Sudden spikes in prices, especially for essential commodities like rice, chili, and shallots, can heavily influence national inflation and public welfare.

In response to this challenge, the Datavidia 9.0 Preliminary Round introduces a data science competition that encourages participants to develop accurate time series forecasting models for food commodity prices across Indonesian provinces. The competition emphasizes main objectives:

1. **Forecast Accuracy** — Participants are required to predict future prices of multiple commodities using various supporting datasets.

Through this competition, participants are expected to enhance their skills in data preprocessing, feature engineering, model development, and scientific communication.

## 📁 Dataset Description

Several datasets were provided to support the forecasting task. Here's a brief overview:

### 1. Food Price Dataset
- **File Name**: Harga Bahan Pangan
- **Period**: January 1, 2022 – December 31, 2024
- **Structure**: 13 commodities × 34 provinces × 1096 days (train set), with predictions required for 92 future days (test set)
- **Objective**: Predict the daily price per commodity for each province during the test period (Oct–Dec 2024)

### 2. Google Trends
- **File Name**: Google Trend
- **Content**: Public interest index (0–100) for each keyword related to food commodities
- **Purpose**: Capture public search behavior as a potential demand indicator

### 3. Currency Exchange Rates
- **File Name**: Mata Uang
- **Included Currencies**: USDIDR, MYRUSD, SGDUSD, THBUSD
- **Purpose**: Monitor currency impact on imported food commodities

### 4. Global Commodity Prices
- **File Name**: Global Commodity Price
- **Content**: Futures data for Crude Oil, Natural Gas, Coal, Palm Oil, Sugar, and Wheat
- **Purpose**: Serve as macroeconomic indicators related to food production and transportation

All data were provided in CSV format and required preprocessing for integration and modeling.


## 🧠 My Approach & Model Details

For this forecasting task, I implemented a deep learning approach using a Long Short-Term Memory (LSTM) model, which is well-suited for handling time series data and learning temporal patterns over long sequences.

Key details of my model:

- **Model Type**: LSTM (Long Short-Term Memory)
- **Look-back Window**: 360 days
- **Epochs**: 200
- **Batch Size**: 128
- **Data Normalization**: MinMax Scaling
- **Input Shape**: Sliding window sequences with shape `(360, n_features)`
- **Output**: One-step-ahead price prediction for each day in the test period

I experimented with several variations and hyperparameters, but this configuration provided a stable and reasonably accurate performance across all provinces and commodities. The LSTM model was trained separately for each commodity-province pair.

## 📊 Evaluation Metric and Submission

Submissions were evaluated using **Mean Absolute Percentage Error (MAPE)**, which measures the average percentage difference between predicted values and actual (ground truth) values:

```python
import numpy as np

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```
Or, more simply, with scikit-learn:

```python
from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_true, y_pred)
```
### Why MAPE?
MAPE is intuitive and scale-independent — ideal for evaluating price predictions across different commodities with different price ranges. A lower MAPE indicates a better model performance, which is especially important in high-impact economic contexts like food pricing.

## 📝 Submission Format
Participants were required to submit a .csv file with predicted values for 40,664 rows — corresponding to 13 food commodities × 34 provinces × 92 days.

Scores were shown for both Public and Private leaderboard splits.

My scores:

✅ Private MAPE Score: 0.07445

✅ Public MAPE Score: 0.07471

## 🔒 Final Note & Disclaimer

This repository was created solely for personal learning and documentation purposes as part of my participation in the **Datavidia 9.0 Preliminary Round (2025)**. Please do not directly copy or redistribute the contents without proper credit or permission.

Any similarity in approach or code structure to other participants' work is purely coincidental. The model, data handling, and scripts presented here reflect my individual effort and understanding at the time of submission.

---

© 2025 Kelvin Jonathan Yusach  
All rights reserved.  


