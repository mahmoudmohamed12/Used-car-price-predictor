# 🏎️ Used Car Price Predictor

This project implements a high-precision Machine Learning pipeline to estimate the market value of used Audi vehicles. By leveraging **Gradient Boosting (LightGBM)** and custom feature engineering, the model achieves a high degree of accuracy, explaining **~96%** of the price variance.



## 📊 Performance Summary
* **R² Score:** 0.9601
* **MAE (Mean Absolute Error):** £1,135.90
* **MAPE (Relative Error):** 7.16%
* **Best Model:** LightGBM (Tuned via Optuna)

---

## 🛠️ The Pipeline
The project utilizes a specialized dual-stream `ColumnTransformer` to handle different model requirements, ensuring maximum accuracy for tree-based architectures.

### 1. Feature Engineering
Created synthetic features to capture car wear and efficiency:
* **`mileage_per_year`**: Captures how aggressively the car was driven.
* **`mpg_per_engine`**: Measures fuel efficiency relative to power.
* **`age`**: Derived from the year of manufacture.

### 2. Preprocessing Streams
* **Boosting Stream:** SimpleImputer (Median) ➔ **OrdinalEncoder** (for non-linear efficiency).
* **Final Model:** LightGBM Regressor wrapped within a sklearn Pipeline.

---

## 🧠 Advanced Hyperparameter Tuning: Optuna

I implemented **Optuna** for Bayesian optimization to intelligently search the hyperparameter space of the `LightGBM` model.



### 🛠️ Optimization Strategy
Instead of testing combinations randomly, Optuna builds a probability model to select new hyperparameters based on previous trial performance.

* **Bayesian Search:** Focuses on promising regions of the parameter space by learning from previous iterations.
* **Pruning:** Automatically terminates poorly performing trials early to focus resources on high-potential candidates.

### 📈 Generalization Check
The gap between training performance and test performance is a critical health indicator for any AI model.

| Metric | Value |
| :--- | :--- |
| **Cross-Validation R²** | 0.9631 |
| **Test Set R²** | 0.9601 |
| **Delta** | `0.0030` |

> **Verdict:** The extremely small difference ($<0.01$) indicates that the model **strong generalization with minimal overfitting.** to unseen data.

---

## 🚀 How to Use (Interactive Web App)

A **Streamlit** web application was developed for easy deployment and usage.

1.  **Install dependencies:**
    ```bash
    pip install streamlit pandas joblib lightgbm scikit-learn
    ```
2.  **Run the App:**
    ```bash
    streamlit run app.py
    ```



---
## 🧰 Tech Stack
- Python
- Pandas & NumPy
- Scikit-learn
- LightGBM
- Optuna (Bayesian Optimization)
- Streamlit
- Matplotlib
---

## 🎯 Business Impact

This model can assist:
- Dealerships in pricing inventory accurately
- Buyers in evaluating fair market value
- Financial institutions in risk-based vehicle valuation
---


## 📂 Project Structure
* `data/`: Raw Audi dataset.
* `app.py`: Streamlit application code.
* `car_price_predictor_pipeline.pkl`: Saved model pipeline.
* `requirements.txt`: List of dependencies.
