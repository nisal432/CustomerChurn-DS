# Data_scicence
Data_science project (Customer chern)
Customer churn is a process where customer leave or stop using your company products.
It is very useful as nowadays it is very necessary for the businesses to retain their costumers and to know if the customers are most likely to leave and find the reason.

With the help of this model we will know customers that are potentially at risk and help to prevent them from leaving possibly saving your business.

Components of the churn models are 
-data collection about the customer (all the historical data transaction, interaction with       website or app and their feedback
-Feature engineering (you identify and create new features that can influence churn, -it can be demographics account balance, interaction with the product.)
-Model Selection (logistic regression, decision trees, random forest, gradiate boosting)
-Training and evaluation (training model and evaluating performance using accuracy precision F1 score, Aucrocove)
-Deployment and monitoring


Best services:
-Subscription based companies
-Telecommunication and internet provider
-Banking and financial service provider


Data used in this project:
https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling?resource=download
currently building @14:12


## 📋 Project Overview

This project aims to predict whether a bank customer will **churn** (leave the bank) using machine learning. Customer churn prediction helps banks identify at-risk customers and take proactive retention measures.

The model is built on the **Churn_Modelling** dataset and includes data exploration, feature engineering, multiple ML models, and performance comparison.

---

## 📊 Dataset

- **Dataset Name**: Churn_Modelling.csv
- **Rows**: 
- **Target Variable**: `Exited` (1 = Customer churned, 0 = Stayed)
- **Key Features**:
  - CreditScore, Age
  - HasCrCard, IsActiveMember, EstimatedSalary
  - Geography 

---

## 🔧 Feature Engineering

The following new features were created to improve model performance:


---

## 📈 Models Evaluated

| Model                        | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
|-----------------------------|----------|-------------------|----------------|------------------|
| **Random Forest**           | **0.8695** | 0.76              | 0.49           | 0.59             |
| **Logistic Regression**     | 0.8110   | 0.55              | 0.20           | 0.29             |

### Best Model: **Random Forest Classifier**

**Why Random Forest?**
- Highest accuracy (86.95%)
- Good balance of precision and recall
- Robust and interpretable (feature importance available)
- Handles imbalanced data relatively well

---

## 🛠 Technologies Used

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (all models + preprocessing)
- Jupyter Notebook

---

## 🚀 How to Run the Project

### 1. Clone or Download the Project

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
3. Run the Notebook
Open projecttest.ipynb in Jupyter Notebook or Jupyter Lab:
Bashjupyter notebook
4. Key Steps in the Notebook

Load and explore data
Data cleaning & preprocessing
Feature engineering
Train multiple models
Evaluate and compare performance
Feature importance analysis


📊 Key Insights

Customers with low product usage and zero balance are more likely to churn.
Age and Geography play significant roles.
Tree-based models (Random Forest & Gradient Boosting) significantly outperform linear models.
The churn class is imbalanced — recall for churners is the main area for improvement.


🔮 Future Improvements

Handle class imbalance using SMOTE or class weights
Hyperparameter tuning with GridSearchCV / Optuna
Try advanced models (XGBoost, LightGBM, CatBoost)
Deploy model using Flask / FastAPI or Streamlit
Add SHAP values for better interpretability
Customer segmentation + personalized retention strategies


📄 License
This project is for educational and portfolio purposes.

👤 Author
Built as part of a Machine Learning project.
