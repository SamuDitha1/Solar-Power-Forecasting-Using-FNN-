# ☀️ Hourly Solar PV Power Forecasting using Feedforward Neural Networks (FNN)

This repository contains an implementation of a **Feedforward Neural Network (FNN)** for **hour-ahead solar PV power forecasting**.  
The model uses **advanced time-series feature engineering** to compensate for the lack of inherent memory in FNNs and achieves **high prediction accuracy (>97%)**.

This work is based on the accompanying research paper:
**“Hourly Solar Generation Forecasting With Feedforward Neural Networks”**

---

## 📌 Project Objectives

- Accurately forecast **next-hour solar PV power output**
- Reduce uncertainty in solar generation
- Support grid operation, load balancing, and renewable integration
- Provide a **computationally efficient alternative** to LSTM/GRU models

---

## 🧠 Key Features

- Feedforward Neural Network (FNN) with **residual connection**
- Extensive **time-series feature engineering**
- Robust data preprocessing and normalization
- Early stopping and adaptive learning rate
- Automatic result visualization and metric reporting
- Optional **Tkinter GUI** for result inspection

---

## 🏗️ Model Architecture

| Layer | Neurons | Activation | Dropout |
|-----|--------|------------|---------|
| Input | 137 features | – | – |
| Hidden 1 | 1024 | Swish | 0.30 |
| Hidden 2 | 512 | Swish | 0.25 |
| Hidden 3 | 256 | Swish + Residual | 0.20 |
| Hidden 4 | 128 | Swish | 0.15 |
| Hidden 5 | 64 | Swish | 0.10 |
| Hidden 6 | 32 | Swish | 0.05 |
| Output | 1 | Linear | – |

- **Loss function**: Huber Loss  
- **Optimizer**: Adam  
- **Seed**: 42 (reproducibility)

---

## 🛠️ Feature Engineering

### Temporal Features
- Hour, day of week, day of year, month, week
- Weekend and seasonal flags
- Morning / peak / evening solar ramps

### Cyclical Encoding
- `sin` / `cos` encoding for hour, day, month, year

### Solar-Specific Feature
- Half-sine solar position curve (06:00–18:00)

### Historical Memory Features
- Lag features: up to **72 hours**
- Rolling statistics: mean, std, min, max, median
- Exponential Moving Averages (EMA): α = 0.1, 0.3, 0.5
- Differencing and momentum features

---



