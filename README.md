# Error-Detection-in-Radiation-Oncology

# Interpretable Treatment Plan Review AI (Radiation Oncology)

This repository contains a machine learning pipeline that explores **interpretable AI assistance for radiation oncology treatment plan review**. The project compares **decision-tree–based classifiers** (XGBoost, LightGBM, and an ensemble) against an **autoencoder anomaly-detection approach**, with a focus on both **performance** (precision/recall/F1) and **interpretability** (latent-space visualization and per-feature reconstruction errors).  
(See project paper for full context and results.) 

## Project Motivation

Radiation oncology treatment plans are **error-prone** and require **manual, time-consuming review** before clinical deployment. This project investigates whether domain-specific ML can help **flag abnormal plans** and provide **interpretable signals** about which plan parameters may be erroneous—supporting faster and safer review workflows.

## Methodologies

1. **Loads and preprocesses treatment-plan data** (numeric plan parameters + tumor location as categorical, when applicable).
2. **Generates synthetic “abnormal” plans** by perturbing normal plans using a controlled error-introduction algorithm (sampling + feature-wise perturbation rules).
3. Trains and evaluates:
   - **Autoencoder** on numeric features (anomaly detection via reconstruction error).
   - **XGBoost** classifier.
   - **LightGBM** classifier.
   - **Ensemble** combining XGBoost + LightGBM + a small focal-loss neural net (FocalNet-style MLP).
4. Provides **interpretability outputs**:
   - **2D/3D latent space visualizations** (autoencoder encoder output).
   - **Per-feature reconstruction comparisons** for individual plans.
   - Reconstruction-error thresholding to flag anomalies.

## Data

- The dataset is **not public** and was obtained via Boston Medical Center (BMC) in partnership with Boston University (BU).  
- The pipeline assumes:
  - Base dataset = **normal plans**
  - Abnormal plans = generated synthetically using an error-introduction algorithm
- Features described in the paper include:
  - Tumor location (categorical)
  - Numeric plan parameters such as: `ptv_dose_rx`, `rx_count`, `dose_per_fraction`, `total_fractions`, `beam_count`, `gantry_angle`, `mu_per_deg`, `mu_per_cgy`

### Data Access (as referenced in the paper)
If you have access permissions, the paper references these links:

```text
Colab notebook:
https://colab.research.google.com/drive/19lee5MqAdBi1M9GwxsrhF1HuIKne3k9z?usp=sharing

Dataset (restricted access):
https://drive.google.com/file/d/18jcAEnjyf3BSWbcLomuzm4O8LULqz-oH/view?usp=share_link
