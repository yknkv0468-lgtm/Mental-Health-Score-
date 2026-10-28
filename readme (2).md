# 🧠 Student Social Media & Mental Health Impact: AI Diagnostic Suite

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An interactive Machine Learning dashboard designed to analyze and predict the impact of social media habits, screen time, and lifestyle factors on student mental health. Powered by a highly accurate **Random Forest Regressor** (R² = ~89%), this application serves as both a diagnostic tool and a "what-if" habit simulator for digital wellness.

## 🚀 Key Features

* **🎯 Live Prediction & Diagnosis:** Input student demographics, screen time limits, and sleep schedules to receive a real-time Mental Health Score prediction alongside AI-driven wellness recommendations.
* **🔄 "What-If" Habit Simulator:** Interactively adjust specific habits (e.g., daily unlocks, sleep hours) on a live sensitivity curve to visualize how behavioral changes directly impact projected mental well-being.
* **📊 Exploratory Data Analysis (EDA):** A comprehensive, interactive data explorer showcasing correlations, distributions, and platform usage trends from a cohort of 5,000 students.
* **⚙️ Model Architecture & Benchmarks:** Transparent insights into the Scikit-Learn `ColumnTransformer` preprocessing pipeline and model feature importance.

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (Pipelines, Ensembles, Preprocessing)
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Plotly Express, Plotly Graph Objects
* **Model Serialization:** Joblib

## 📂 Repository Structure

```text
├── app.py                                          # Main Streamlit application script
├── compress.py                                     # Utility script for local model training & compression
├── requirements.txt                                # Python dependencies for deployment
├── Mental_Health_Model.pkl                         # Serialized Random Forest model (compressed)
├── Student Social Media And Mental Health Impact.csv # Dataset (5,000 records)
└── README.md                                       # Project documentation
```

## 💻 Local Installation & Setup

To run this project locally on your machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/mental-health-ml-app.git
   cd mental-health-ml-app
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   The application will launch in your default web browser at `http://localhost:8501`.

## 🧠 Machine Learning Architecture

The core of this application is a supervised regression pipeline built using `scikit-learn`. 

* **Preprocessing:** A unified `ColumnTransformer` handles:
  * **Skewed Numeric Data:** Logarithmic transformation (`np.log1p`) followed by Standard Scaling.
  * **Standard Numeric Data:** Outlier clipping and Standard Scaling.
  * **Ordinal Data:** Rank-preserving `OrdinalEncoder` (e.g., Stress Levels).
  * **Categorical Data:** Nominal feature grouping and `OneHotEncoder`.
* **Estimator:** `RandomForestRegressor(random_state=42)` 
* **Performance Metrics (Test Set):**
  * **R² Score:** 0.8928
  * **Mean Absolute Error (MAE):** 0.3214
  * **Root Mean Squared Error (RMSE):** 0.4312

## 🌐 Live Deployment

This application is deployed on **Streamlit Community Cloud**. 
You can access the live version here: **[Link to your live app]** *(Note: Replace this link with your actual Streamlit URL once deployed)*.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.