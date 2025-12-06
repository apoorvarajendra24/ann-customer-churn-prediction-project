# ANN Customer Churn Prediction Project

## 📝 Project Overview

This project builds an **Artificial Neural Network (ANN)** model to predict customer churn. Based on a dataset of customer attributes, the model learns to classify whether a customer is likely to leave (churn) or stay. The project covers the end-to-end pipeline — from data loading and preprocessing, through model training and hyperparameter tuning, to deployment and prediction.

---

## 📂 Repository Structure

| File / Folder | Description |
|---------------|-------------|
| `Churn_Modelling.csv` | The raw customer dataset used for training and testing the model. |
| `experiments.ipynb` | Notebook for exploratory data analysis (EDA), data preprocessing, feature engineering, and initial experiments. |
| `hyperparameterTuning.ipynb` | Notebook for hyperparameter tuning of the ANN model to optimize performance. |
| `prediction.ipynb` | Demonstration notebook showing how to use the trained model to predict churn for new data. |
| `model.h5` | Trained ANN model saved in HDF5 format. |
| `one_hot_encoder_geography.pkl`, `label_encoder_gender.pkl`, `scaler.pkl` | Preprocessing artifacts: encoders and scaler used to transform input data before prediction. |
| `app.py` | (Optional) Script for deploying a simple interface (or API) to input customer data and get churn predictions. |
| `requirements.txt` | List of dependencies / Python packages required to run the project. |

---

## 🧰 Tech Stack & Dependencies

- Python  
- Data handling & preprocessing: `pandas`, `numpy`, `scikit-learn`  
- Deep learning: `tensorflow`, `keras`  
- (Optional) For notebooks: `matplotlib`, `seaborn` / other visualization libraries  
- Additional: any other libraries you used (as listed in `requirements.txt`)

---

## 🚀 How to Run / Use the Project

1. Clone the repository  
    ```bash
    git clone https://github.com/apoorvarajendra24/ann-customer-churn-prediction-project.git
    cd ann-customer-churn-prediction-project
    ```

2. Install the required packages  
    ```bash
    pip install -r requirements.txt
    ```

3. (Optional) Run the notebooks to reproduce EDA, preprocessing, model training or predictions:  
    - `experiments.ipynb`  
    - `hyperparameterTuning.ipynb`  
    - `prediction.ipynb`

4. Use the pre-trained model to make predictions:  
    - Load `model.h5` along with the encoder / scaler pickles  
    - Preprocess your input data using the same transformations  
    - Use the model to predict churn (e.g., churn vs. non-churn)  

5. (Optional) Run `app.py` to deploy a simple interface or API for predictions (if implemented).

---

## 📈 What the Model Does

- Trains a neural network to classify customers into “will churn” or “will stay.”  
- Uses preprocessing steps (encoding categorical variables, scaling numerical features) so that input data is transformed consistently.  
- Supports reuse: you can load the trained model + preprocessing artifacts and predict churn status for **new or unseen customers**.  
- (If you choose) Supports hyperparameter tuning to improve model performance, and experimentation via Jupyter Notebooks.

---

## ✅ Why This Project Matters — Use Cases

Customer churn — i.e., customers leaving a service — is a major issue for many businesses (telecom, banking, subscription-based services, etc.). By proactively identifying customers with a high risk of churn, companies can:  

- Offer retention incentives to at-risk customers  
- Personalize marketing or engagement strategies  
- Allocate resources more efficiently (target retention efforts where needed)  

As a result, churn-prediction models can save significant revenue and help improve customer loyalty.

---

## 💡 Possible Next Steps / Improvements

- Add more evaluation metrics beyond accuracy — e.g. precision, recall, F1-score, ROC AUC. This gives a better picture of model performance, especially in imbalanced datasets.  
- Add cross-validation or k-fold validation to ensure model generalizes well.  
- Add feature-importance or interpretability (using SHAP / LIME / similar) to understand which customer features contribute most to churn prediction.  
- Add data-validation checks (missing values, outliers) and data-cleaning steps if needed.  
- If desired, wrap the model into a web app or REST API for real-time predictions (e.g. using Streamlit, Flask, FastAPI).  
- Expand dataset / add new features (customer behavior data, demographics, usage patterns) to potentially improve predictive power.

---
