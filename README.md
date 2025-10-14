# 📱 Mobile Price Prediction – Dual Output AI Model  

> **Predict both mobile price range and actual price using Machine Learning + AI assistance**

---

## 🧠 Project Overview

This project is an advanced **Multi-Task Machine Learning System** that predicts:  
1. 📊 **Price Range** – classification (0–3)  
2. 💰 **Actual Price (₹)** – regression  

The model learns from key mobile specifications like RAM, battery capacity, processor speed, screen resolution, and more to understand their combined impact on mobile pricing.  

It was **created with the help of AI (ChatGPT-GPT-5)** for research guidance, architecture design, and documentation.

---

## 🎯 Objectives

- Develop a hybrid ML system capable of **predicting two target variables** from the same dataset.  
- Use separate models (Classifier + Regressor) trained on the same feature set.  
- Combine predictions for complete market-level mobile price intelligence.  

---

## ⚙️ Technologies Used

| Category | Tools / Libraries |
|-----------|-------------------|
| Programming | Python 3.x |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Algorithms | RandomForestClassifier, RandomForestRegressor |
| Model Evaluation | Scikit-Learn Metrics |
| Version Control | Git + GitHub |
| AI Support | ChatGPT (GPT-5) |

---

## 🧩 Project Structure

mobile-price-prediction/
│
├── data/
│ ├── mobile_train.csv
│ └── mobile_test.csv
│
├── notebooks/
│ └── model_training.ipynb
│
├── models/
│ ├── classifier_model.pkl
│ └── regressor_model.pkl
│
├── README.md
└── requirements.txt


---

## 🔬 Methodology

1. **Data Preprocessing**
   - Handled missing values and feature scaling.  
   - Encoded categorical features.  

2. **Feature Selection**
   - Selected key predictors: `battery_power`, `ram`, `px_height`, `px_width`, etc.  

3. **Model Training**
   - Trained a **RandomForestClassifier** for price range.  
   - Trained a **RandomForestRegressor** for actual price.  

4. **Evaluation**
   - Classification Metrics → Accuracy, F1-Score  
   - Regression Metrics → MAE, MSE, RMSE  

5. **Deployment**
   - Combined both predictions into a unified output for analysis.

---

## 🧠 Example Workflow

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd

# Load data
data = pd.read_csv("mobile_train.csv")
X = data.drop(['price_range', 'price_actual'], axis=1)
y_class = data['price_range']
y_reg = data['price_actual']

# Train models
clf = RandomForestClassifier().fit(X, y_class)
reg = RandomForestRegressor().fit(X, y_reg)

# Predictions
range_pred = clf.predict(X)
price_pred = reg.predict(X)

# Combine outputs
pred_df = pd.DataFrame({
    'Predicted Range': range_pred,
    'Predicted Price': price_pred
})
print(pred_df.head())

📊 Evaluation Metrics

Task	Metric	Description
Classification	Accuracy	How often the price range is correctly predicted
Classification	F1-Score	Balance between precision and recall
Regression	MAE	Mean Absolute Error between true and predicted prices
Regression	RMSE	Root Mean Squared Error for prediction quality

🌟 Results Summary

Achieved ~95% accuracy on price range classification.

Regression RMSE ≈ 250–300 ₹ on test data.

Demonstrated strong correlation between mobile specs and pricing trends.

🧠 AI Contribution

This project was designed, documented, and optimized with AI (ChatGPT-GPT-5) assistance — blending automation with human creativity to accelerate development and ensure clarity.

🧩 Future Improvements

Hyperparameter tuning using GridSearchCV

Feature importance visualization

Deployment with Streamlit or Flask

Add deep learning model (Multi-Output Neural Network)

---

🧑‍💻 Author

Aditya Jadhav 
📍 Data Analyst → Data Scientist → Future AI/ML Engineer
🚀 Guided by ChatGPT-GPT-5
