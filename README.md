# ChurnIQ
### AI-Powered Customer Retention & Intelligence Platform

ChurnIQ is a portfolio-grade customer churn analytics application built with **Python, Streamlit and Scikit-learn**. It helps users explore historical customer behaviour, estimate churn risk, score customer files in batches and review model performance through a simple business-friendly interface.

**Live application:** https://customer-churn-prediction-an.streamlit.app/

---

## Product overview

Customer churn can reduce recurring revenue and increase customer acquisition costs. ChurnIQ demonstrates a machine-learning workflow that estimates the likelihood of a bank customer leaving, then presents the result in a format that can be explored by both technical and non-technical users.

The application is an educational and portfolio demonstration using the included sample banking dataset. It is not a live connection to a bank's customer systems.

## Features

| Workspace | What it provides |
|---|---|
| **Overview** | Dataset size, historical churn count and rate, age-group patterns and customer outcome distribution |
| **Predict Churn** | Individual customer scoring with a probability gauge, outcome label and risk band |
| **Batch Prediction** | CSV upload, input validation, bulk scoring, risk summaries and downloadable results |
| **Customer Explorer** | Filter historical records by geography, age and churn outcome |
| **Model Performance** | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix and classification report |
| **Explainability** | Global permutation feature importance |
| **Reports** | Download dataset summaries and customer data exports |
| **About Model** | Model methodology, input fields and responsible-use information |

## Application preview

The interface uses a minimal light theme with a sidebar workspace, clear metric cards, simple charts and consistent risk labels.

## Machine-learning workflow

1. Read the supplied `Churn_Modelling.csv` dataset.
2. Select **`Exited` explicitly** as the target variable.
3. Remove identifier fields: `RowNumber`, `CustomerId` and `Surname`.
4. Split the dataset into training and test partitions using a stratified 80/20 split and a fixed random seed.
5. Impute missing numeric and categorical values.
6. One-hot encode categorical inputs, with unknown categories handled safely by the preprocessing pipeline.
7. Train a class-balanced Random Forest classifier.
8. Evaluate the held-out predictions and display classification metrics.
9. Use the trained pipeline for single-customer and batch predictions.

### Model configuration

- Algorithm: Random Forest Classifier
- Estimators: 250 trees
- Class weighting: balanced
- Minimum samples per leaf: 2
- Random state: 42
- Test size: 20%
- Prediction threshold: 0.50

The application trains and caches its pipeline at startup. Metrics shown in Model Performance are calculated from the same held-out split used during that startup training run.

### Model audit and correction

The earlier training script selected the last dataframe column after one-hot encoding as the target. Depending on the resulting column order, this could select `Gender_Male` rather than the intended `Exited` churn label. The ChurnIQ pipeline now selects `Exited` by name and excludes it from the feature set.

The older Keras model, scaler and feature-list artifacts remain in the repository for historical reference. The current ChurnIQ application does **not** load those legacy artifacts.

## Technology stack

- **Language:** Python
- **Application:** Streamlit
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Visualisation:** Plotly
- **Model pipeline utilities:** Joblib

## Repository structure

```text
customer-churn-prediction/
├── app.py                       # ChurnIQ Streamlit application
├── train_model.py               # Reproducible model training script
├── customer-churn-prediction.py # Training script retained under the original filename
├── Churn_Modelling.csv          # Sample customer churn dataset
├── requirements.txt             # Python dependencies
├── churn_model.keras            # Legacy Keras model artifact (not used by current app)
├── scaler.joblib                # Legacy scaler artifact (not used by current app)
└── features.joblib              # Legacy feature list (not used by current app)
```

Running `train_model.py` produces `churniq_model.joblib` and `churniq_metrics.joblib`. The Streamlit app currently trains its own in-memory pipeline so that its displayed evaluation metrics and predictions use the same model instance.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/krishna-shanu30/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start ChurnIQ

```bash
streamlit run app.py
```

The application expects `Churn_Modelling.csv` in the repository root.

## Train a reusable model artifact

To run the standalone training workflow:

```bash
python train_model.py
```

This saves:
- `churniq_model.joblib` — fitted preprocessing and classification pipeline.
- `churniq_metrics.joblib` — held-out evaluation metrics.

The application currently retrains its in-memory pipeline on startup rather than loading these saved artifacts.

## Batch prediction CSV format

Use the **Download sample CSV** option in the Batch Prediction workspace as the recommended template.

Required columns:

| Column | Description |
|---|---|
| `CreditScore` | Customer credit score |
| `Geography` | Customer geography (France, Germany or Spain in the sample data) |
| `Gender` | Customer gender category from the sample data |
| `Age` | Customer age |
| `Tenure` | Years as a customer |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products |
| `HasCrCard` | Credit card indicator (0 or 1) |
| `IsActiveMember` | Active member indicator (0 or 1) |
| `EstimatedSalary` | Estimated salary |

The uploader checks required columns, rejects empty files and limits a batch to 100,000 records. Use the original category values in the sample template.

## Understanding churn scores

The model returns a probability between 0 and 1. ChurnIQ presents that probability as a percentage.

| Risk band | Probability | Suggested interpretation |
|---|---:|---|
| Low | Below 30% | Lower estimated churn risk |
| Medium | 30% to below 70% | Monitor and review |
| High | 70% or above | Consider a personalised retention review |

The binary outcome label uses a 50% threshold. Risk bands are communication aids, not calibrated business policies or guarantees.

## Model evaluation

The Model Performance workspace reports:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Confusion matrix
- Classification report
- ROC curve

The Explainability workspace uses permutation importance to show global feature importance. It does not provide causal explanations or guarantee why an individual customer will churn.

## Limitations and responsible use

- This project uses a historical sample banking dataset, not live customer data.
- Performance on this dataset may not generalise to other banks, time periods or customer populations.
- Model probabilities should not be treated as calibrated probabilities without additional calibration analysis.
- Predictions are estimates and should support, not replace, human judgement.
- Do not use this demonstration as the sole basis for credit, pricing, eligibility or other consequential decisions.
- Before real-world deployment, validate data quality, fairness, calibration, security, privacy, monitoring and business impact with representative and appropriately governed data.

## Deployment

The project has an existing Streamlit Community Cloud deployment:

https://customer-churn-prediction-an.streamlit.app/

A deployment may need a rebuild after dependency or application changes. Check the Streamlit deployment logs if startup fails.

## Author

**Krishna Chandak**

AI & ML | Machine Learning | Deep Learning | Data Science
