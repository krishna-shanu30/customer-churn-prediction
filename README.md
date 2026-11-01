# ChurnIQ — AI-Powered Customer Retention Intelligence

ChurnIQ is a customer churn analytics and prediction application built with Python, Streamlit, Pandas, Scikit-learn and Plotly. It is designed as both a portfolio demonstration and a clear business-facing prototype.

**Live demo:** https://customer-churn-prediction-an.streamlit.app/

## Features
- **Overview:** customer base, historical churn rate and age-based churn patterns.
- **Predict Churn:** estimate an individual customer's churn probability and risk band.
- **Batch Prediction:** upload a customer CSV, score records and download results.
- **Customer Explorer:** filter the supplied dataset by geography, age and historical outcome.
- **Model Performance:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix and classification report.
- **Explainability:** permutation feature importance for global model drivers.
- **Reports:** download dataset summaries and filtered customer data.
- **About Model:** inputs, risk thresholds and responsible-use notes.

## Model methodology
The application explicitly uses `Exited` as the prediction target and removes row number, customer ID and surname from model inputs. It uses a preprocessing pipeline with numeric imputation, one-hot encoding and a class-balanced Random Forest classifier.

Evaluation uses a stratified 80/20 train-test split with a fixed random seed. The app trains and caches its model at startup so displayed metrics correspond to the model currently in use.

**Model audit note:** the original training script selected the final dataframe column as its target after one-hot encoding. This could select `Gender_Male` rather than `Exited`. The ChurnIQ app avoids this issue by selecting `Exited` explicitly. Legacy Keras model files are retained for traceability but are not used by the new app.

## Run locally
1. Clone the repository and create a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Start the app: `streamlit run app.py`

The app reads `Churn_Modelling.csv` from the repository directory.

## Train and save a reusable model
Run `python train_model.py` to create `churniq_model.joblib` and `churniq_metrics.joblib`. The Streamlit app currently trains and caches its own in-memory pipeline to keep the displayed evaluation metrics aligned with its holdout split.

## Batch CSV input
Use the sample CSV downloaded from the Batch Prediction page. Required columns are:

`CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary`

Use original category values for Geography and Gender.

## Risk interpretation
- **Low:** probability below 30%.
- **Medium:** 30% to below 70%.
- **High:** 70% or above.

The binary prediction threshold is 50%. Risk bands are communication aids, not calibrated business policies.

## Responsible use
This is an educational portfolio demonstration built on a sample banking dataset. Predictions are estimates, not guarantees or causal explanations. Do not use this demo as the sole basis for customer treatment, credit decisions or other consequential actions. Validate on representative, current and appropriately governed data before any real deployment.

## Technology
Python · Streamlit · Pandas · NumPy · Scikit-learn · Plotly · Joblib

## Author
Krishna Chandak
