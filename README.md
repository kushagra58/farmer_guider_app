# 👨‍🌾 Farmer Guider AI App (Unsupervised Learning)

An interactive AI web application built with Python and Flask that groups farmers sharing similar microclimates and soil characteristics using unsupervised machine learning. By utilizing **K-Means Clustering**, the app segmentizes regional environmental profiles and serves actionable, data-driven farming recommendations.

## 🎯 Project Objective
The goal is to build a clustering system that segments farmers based purely on input environmental attributes (without relying on crop target labels). The application uses these profiles to output:
* 💧 Custom irrigation workflows.
* 🌱 Precise fertilizer balancing advice.
* 🚜 Long-term soil management blueprints.

---

## 📊 Dataset Profile
This project utilizes the **Crop Recommendation Dataset**. To enforce true unsupervised profiling, the target column (`label` / crop name) is explicitly removed prior to data preprocessing and training. 

The machine learning model clusters profiles using the following input features:
* **N, P, K:** Soil nutrient ratios (Nitrogen, Phosphorus, Potassium).
* **Temperature:** Ambient temperature in °C.
* **Humidity:** Relative humidity percentage.
* **pH:** Soil acidity/alkalinity measurement.
* **Rainfall:** Average rainfall volume in mm.

---

## ⚙️ Architecture & Implementation Requirements
* **Data Exploration:** Automated structural checking and handling of missing values via feature column averages.
* **Feature Engineering:** Data scaling executed via `StandardScaler` to handle variances across units (e.g., pH vs Rainfall).
* **Optimization:** Evaluation of the optimal number of clusters ($K$) using the WCSS Elbow Method and structural valuation via the **Silhouette Score**.
* **Visual Verification:** Matplotlib-driven data plots mapping clusters relative to environmental thresholds.
* **Deployment:** Web service interface driven by Flask with a responsive user interface allowing real-time profiling predictions.

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
Ensure you have Python 3.8+ installed along with `pip`. 

### 2. Setup
```bash

# Install package dependencies
pip install flask pandas scikit-learn matplotlib numpy
