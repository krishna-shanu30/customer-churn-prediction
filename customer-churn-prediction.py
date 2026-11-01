"""Train ChurnIQ with Exited explicitly selected as the target."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT=Path(__file__).resolve().parent
TARGET="Exited"
DROP=["RowNumber","CustomerId","Surname"]

def main():
    data=pd.read_csv(ROOT/"Churn_Modelling.csv")
    if TARGET not in data.columns:
        raise ValueError(f"Dataset must contain {TARGET!r}.")
    X=data.drop(columns=DROP+[TARGET],errors="ignore")
    y=data[TARGET].astype(int)
    numeric=X.select_dtypes(include=["number"]).columns.tolist()
    categorical=X.select_dtypes(exclude=["number"]).columns.tolist()
    preprocess=ColumnTransformer([
        ("numeric",SimpleImputer(strategy="median"),numeric),
        ("categorical",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                                 ("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)
    ])
    model=Pipeline([("preprocess",preprocess),("classifier",RandomForestClassifier(
        n_estimators=250,class_weight="balanced",min_samples_leaf=2,random_state=42,n_jobs=-1))])
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    model.fit(X_train,y_train)
    probabilities=model.predict_proba(X_test)[:,1]
    predictions=(probabilities>=.5).astype(int)
    metrics={"accuracy":accuracy_score(y_test,predictions),
        "precision":precision_score(y_test,predictions,zero_division=0),
        "recall":recall_score(y_test,predictions,zero_division=0),
        "f1":f1_score(y_test,predictions,zero_division=0),
        "roc_auc":roc_auc_score(y_test,probabilities)}
    joblib.dump(model,ROOT/"churniq_model.joblib")
    joblib.dump(metrics,ROOT/"churniq_metrics.joblib")
    print("Target:",TARGET)
    print("Features:",list(X.columns))
    for key,value in metrics.items(): print(f"{key}: {value:.4f}")
    print("Saved churniq_model.joblib and churniq_metrics.joblib")

if __name__=="__main__":
    main()
