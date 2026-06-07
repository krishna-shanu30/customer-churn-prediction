# Customer Churn Prediction

A Deep Learning-based web application that predicts whether a customer is likely to leave a bank (customer churn) using customer demographic and banking information.

Live Demo:
https://customer-churn-prediction-an.streamlit.app/

## Features:
1. Predict customer churn probability in real time
2. Single customer prediction through an interactive web interface
3. Batch prediction using CSV file upload
4. Risk level classification (Low, Medium, High)
5. Download prediction results as CSV
6. Built with Streamlit for a user-friendly experience

## Technologies Used:
1. Python
2. TensorFlow / Keras
3. Streamlit
4. Pandas
5. NumPy
6. Scikit-learn
7. Joblib

## How It Works:
1. User enters customer information.
2. Input data is preprocessed and scaled.
3. The trained neural network model predicts churn probability.
4. Results are displayed with:
   1. Churn Probability
   2. Prediction (Will Stay / Will Churn)
   3. Risk Level
  
## Batch Prediction:

Users can upload a CSV file containing multiple customer records. The application:

1. Processes all records
2. Predicts churn probability
3. Generates churn labels
4. Allows downloading results as CSV

## Future Improvements:
1. Explainable AI (XAI) integration
2. Customer retention recommendations
3. Interactive dashboards
4. Advanced model comparison
5. API deployment using Flask/FastAPI
6. Cloud database integration

## Author

Krishna Chandak
