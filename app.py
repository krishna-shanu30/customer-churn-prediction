from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT=Path(__file__).resolve().parent
DATA_PATH=ROOT/"Churn_Modelling.csv"
TARGET="Exited"
DROP_COLUMNS=["RowNumber","CustomerId","Surname"]

st.set_page_config(page_title="ChurnIQ | Customer Intelligence",page_icon="📊",layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#F7F8FA}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #E8EBF0}
[data-testid="stMetric"]{background:#fff;padding:16px 18px;border:1px solid #E8EBF0;border-radius:13px}
.hero{background:#fff;border:1px solid #E8EBF0;padding:24px 28px;border-radius:16px;margin-bottom:18px}
.hero h1{color:#172554;font-size:2rem;margin:0 0 5px}.hero p{color:#64748B;margin:0}
div.stButton>button,div.stDownloadButton>button{border-radius:9px;font-weight:600}
</style>""",unsafe_allow_html=True)

@st.cache_data
def load_data():
    if not DATA_PATH.exists(): raise FileNotFoundError("Churn_Modelling.csv was not found.")
    data=pd.read_csv(DATA_PATH)
    if TARGET not in data: raise ValueError("Dataset must include the Exited target column.")
    return data

@st.cache_resource
def train_model():
    data=load_data()
    X=data.drop(columns=DROP_COLUMNS+[TARGET],errors="ignore")
    y=data[TARGET].astype(int)
    numeric=X.select_dtypes(include=["number"]).columns.tolist()
    categorical=X.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_pipe=Pipeline([("imputer",SimpleImputer(strategy="median"))])
    categorical_pipe=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown="ignore"))])
    prep=ColumnTransformer([("numeric",numeric_pipe,numeric),("categorical",categorical_pipe,categorical)])
    pipeline=Pipeline([("preprocess",prep),("classifier",RandomForestClassifier(
        n_estimators=250,class_weight="balanced",min_samples_leaf=2,
        random_state=42,n_jobs=-1))])
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    pipeline.fit(X_train,y_train)
    probs=pipeline.predict_proba(X_test)[:,1]
    pred=(probs>=.5).astype(int)
    metrics={"accuracy":accuracy_score(y_test,pred),"precision":precision_score(y_test,pred,zero_division=0),
        "recall":recall_score(y_test,pred,zero_division=0),"f1":f1_score(y_test,pred,zero_division=0),
        "auc":roc_auc_score(y_test,probs),"y_test":y_test.to_numpy(),"probabilities":probs,
        "predictions":pred,"confusion":confusion_matrix(y_test,pred),
        "report":classification_report(y_test,pred,output_dict=True,zero_division=0)}
    return pipeline,metrics,numeric,categorical

def hero(title,subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',unsafe_allow_html=True)

def risk(p):
    return "High" if p>=.70 else "Medium" if p>=.30 else "Low"

def probability_gauge(p):
    fig=go.Figure(go.Indicator(mode="gauge+number",value=p*100,number={"suffix":"%"},
      gauge={"axis":{"range":[0,100]},"bar":{"color":"#3157A4"},
      "steps":[{"range":[0,30],"color":"#DDF4E7"},{"range":[30,70],"color":"#FFF0CC"},{"range":[70,100],"color":"#FCE1E1"}]}))
    fig.update_layout(height=220,margin=dict(l=20,r=20,t=25,b=0),paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)

def main():
    try:
        data=load_data()
        model,metrics,numeric,categorical=train_model()
    except Exception as exc:
        st.error(f"ChurnIQ could not initialise: {exc}")
        st.stop()
    with st.sidebar:
        st.markdown("## ChurnIQ")
        st.caption("AI-powered customer retention intelligence")
        page=st.radio("WORKSPACE",["Overview","Predict Churn","Batch Prediction","Customer Explorer","Model Performance","Explainability","Reports","About Model"])
        st.divider()
        st.caption("Random Forest · 250 trees")
        st.caption("Decision threshold: 50%")
    if page=="Overview":
        hero("Customer intelligence, made clearer.","Understand churn risk, explore customer patterns and turn predictions into retention actions.")
        n=len(data); churn=int(data[TARGET].sum())
        a,b,c,d=st.columns(4)
        a.metric("Customers",f"{n:,}"); b.metric("Historical churn",f"{churn:,}")
        c.metric("Historical churn rate",f"{churn/n:.1%}"); d.metric("Holdout ROC-AUC",f"{metrics['auc']:.3f}")
        left,right=st.columns(2)
        with left:
            st.subheader("Churn by age group")
            tmp=data.copy(); tmp["Age group"]=pd.cut(tmp.Age,[0,25,35,45,55,120],labels=["18–25","26–35","36–45","46–55","56+"])
            chart=tmp.groupby("Age group",observed=False)[TARGET].mean().reset_index()
            fig=px.bar(chart,x="Age group",y=TARGET,color_discrete_sequence=["#3157A4"],labels={TARGET:"Churn rate"})
            fig.update_layout(yaxis_tickformat=".0%",plot_bgcolor="white",paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
        with right:
            st.subheader("Historical customer outcomes")
            chart=pd.DataFrame({"Outcome":["Stayed","Churned"],"Customers":[n-churn,churn]})
            fig=px.pie(chart,names="Outcome",values="Customers",hole=.62,color="Outcome",color_discrete_map={"Stayed":"#3157A4","Churned":"#E58A65"})
            st.plotly_chart(fig,use_container_width=True)
        st.caption("Dataset insights are historical and refer only to the supplied sample bank dataset.")
    elif page=="Predict Churn":
        hero("Predict customer churn","Enter a customer's profile to estimate churn probability and risk category.")
        with st.form("customer_form"):
            c1,c2,c3=st.columns(3)
            with c1:
                credit=st.number_input("Credit score",300,850,650); age=st.number_input("Age",18,100,35)
                tenure=st.number_input("Tenure (years)",0,10,5); geography=st.selectbox("Geography",["France","Germany","Spain"])
            with c2:
                balance=st.number_input("Account balance",0.0,300000.0,100000.0,step=1000.0)
                products=st.selectbox("Number of products",[1,2,3,4]); card=st.selectbox("Has credit card?",["Yes","No"])
                active=st.selectbox("Active member?",["Yes","No"])
            with c3:
                salary=st.number_input("Estimated salary",0.0,200000.0,50000.0,step=1000.0)
                gender=st.selectbox("Gender",["Female","Male"])
            submitted=st.form_submit_button("Calculate churn risk",type="primary",use_container_width=True)
        if submitted:
            customer=pd.DataFrame([{"CreditScore":credit,"Age":age,"Tenure":tenure,"Balance":balance,
                "NumOfProducts":products,"HasCrCard":int(card=="Yes"),"IsActiveMember":int(active=="Yes"),
                "EstimatedSalary":salary,"Geography":geography,"Gender":gender}])
            try:
                p=float(model.predict_proba(customer)[0,1])
                left,right=st.columns(2)
                with left: probability_gauge(p)
                with right:
                    st.metric("Predicted outcome","Likely to churn" if p>=.5 else "Likely to stay")
                    st.metric("Risk category",risk(p))
                    if p>=.7: st.warning("High risk: consider a personalised retention review.")
                    elif p>=.3: st.info("Medium risk: monitor engagement and review customer needs.")
                    else: st.success("Low risk based on this model score.")
                st.caption("This is a model estimate, not a guarantee that a customer will leave.")
            except Exception as exc: st.error(f"Prediction failed: {exc}")
    elif page=="Batch Prediction":
        hero("Batch customer scoring","Upload customer records, score them together and export prioritised results.")
        template=pd.DataFrame([{"CreditScore":650,"Geography":"France","Gender":"Female","Age":35,"Tenure":5,"Balance":100000,"NumOfProducts":1,"HasCrCard":1,"IsActiveMember":1,"EstimatedSalary":50000}])
        st.download_button("Download sample CSV",template.to_csv(index=False).encode(),"churniq_sample.csv","text/csv")
        uploaded=st.file_uploader("Upload customer CSV",type=["csv"])
        if uploaded:
            try:
                raw=pd.read_csv(uploaded)
                required=set(numeric+categorical)
                missing=sorted(required-set(raw.columns))
                if missing: st.error("Missing columns: "+", ".join(missing)); st.stop()
                if raw.empty: st.warning("The file contains no customer rows."); st.stop()
                if len(raw)>100000: st.error("Please upload no more than 100,000 records at once."); st.stop()
                probs=model.predict_proba(raw)[:,1]
                result=raw.copy(); result["Churn_Probability"]=(probs*100).round(2)
                result["Predicted_Churn"]=np.where(probs>=.5,"Yes","No")
                result["Risk_Level"]=[risk(float(p)) for p in probs]
                a,b,c,d=st.columns(4)
                a.metric("Scored customers",f"{len(result):,}")
                b.metric("High risk",int((result.Risk_Level=="High").sum()))
                c.metric("Medium risk",int((result.Risk_Level=="Medium").sum()))
                d.metric("Low risk",int((result.Risk_Level=="Low").sum()))
                st.dataframe(result,use_container_width=True,hide_index=True)
                st.download_button("Download predictions",result.to_csv(index=False).encode(),"churniq_predictions.csv","text/csv",type="primary")
            except Exception as exc: st.error(f"Could not process the uploaded file: {exc}")
    elif page=="Customer Explorer":
        hero("Customer explorer","Filter the source dataset to explore customer profiles and historical outcomes.")
        a,b,c=st.columns(3)
        geos=a.multiselect("Geography",sorted(data.Geography.unique()),default=sorted(data.Geography.unique()))
        outcome=b.selectbox("Historical outcome",["All","Stayed","Churned"])
        ages=c.slider("Age range",int(data.Age.min()),int(data.Age.max()),(18,70))
        view=data[data.Geography.isin(geos)&data.Age.between(*ages)]
        if outcome=="Stayed": view=view[view[TARGET]==0]
        elif outcome=="Churned": view=view[view[TARGET]==1]
        st.caption(f"{len(view):,} matching customers")
        cols=["CustomerId","CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","IsActiveMember",TARGET]
        st.dataframe(view[cols],use_container_width=True,hide_index=True)
        st.download_button("Export filtered customers",view.to_csv(index=False).encode(),"churniq_customers.csv","text/csv")
    elif page=="Model Performance":
        hero("Model performance","Metrics calculated on a stratified, held-out test dataset.")
        st.caption("80/20 train-test split · stratified target · random_state=42 · decision threshold=50%.")
        a,b,c,d,e=st.columns(5)
        a.metric("Accuracy",f"{metrics['accuracy']:.1%}"); b.metric("Precision",f"{metrics['precision']:.1%}")
        c.metric("Recall",f"{metrics['recall']:.1%}"); d.metric("F1 score",f"{metrics['f1']:.1%}"); e.metric("ROC-AUC",f"{metrics['auc']:.3f}")
        left,right=st.columns(2)
        with left:
            st.subheader("Confusion matrix")
            fig=px.imshow(metrics["confusion"],text_auto=True,x=["Predicted stay","Predicted churn"],y=["Actual stay","Actual churn"],color_continuous_scale="Blues")
            st.plotly_chart(fig,use_container_width=True)
        with right:
            st.subheader("Classification report")
            report=classification_report(metrics["y_test"],metrics["predictions"],output_dict=True,zero_division=0)
            st.dataframe(pd.DataFrame(report).T.round(3),use_container_width=True)
        fpr,tpr,_=roc_curve(metrics["y_test"],metrics["probabilities"])
        fig=px.area(x=fpr,y=tpr,labels={"x":"False positive rate","y":"True positive rate"},title=f"ROC curve · AUC {metrics['auc']:.3f}")
        fig.add_shape(type="line",x0=0,y0=0,x1=1,y1=1,line=dict(dash="dash",color="#94A3B8"))
        st.plotly_chart(fig,use_container_width=True)
    elif page=="Explainability":
        hero("Explainability","See which input variables are most influential in the trained model.")
        st.write("Permutation importance measures how much held-out model performance changes when each feature is shuffled.")
        from sklearn.inspection import permutation_importance
        if st.button("Calculate feature importance",type="primary"):
            with st.spinner("Calculating permutation importance..."):
                raw=data.drop(columns=DROP_COLUMNS+[TARGET],errors="ignore")
                _,x_test,_,y_test=train_test_split(raw,data[TARGET].astype(int),test_size=.2,random_state=42,stratify=data[TARGET].astype(int))
                result=permutation_importance(model,x_test,y_test,n_repeats=5,random_state=42,scoring="roc_auc",n_jobs=-1)
                imp=pd.DataFrame({"Feature":raw.columns,"Importance":result.importances_mean}).sort_values("Importance").tail(12)
            fig=px.bar(imp,x="Importance",y="Feature",orientation="h",color_discrete_sequence=["#3157A4"])
            st.plotly_chart(fig,use_container_width=True)
            st.caption("Global permutation importance indicates predictive association, not causation or an explanation of an individual decision.")
    elif page=="Reports":
        hero("Reports and exports","Download summaries and customer data for further analysis.")
        summary=data.describe(include="all").T.reset_index(names="Field")
        st.dataframe(summary,use_container_width=True)
        st.download_button("Download dataset summary",summary.to_csv(index=False).encode(),"churniq_summary.csv","text/csv")
        st.download_button("Download source dataset",data.to_csv(index=False).encode(),"churniq_dataset.csv","text/csv")
        st.caption("Review data permissions before sharing exported customer information.")
    else:
        hero("About ChurnIQ","An AI-powered customer retention intelligence demonstration.")
        st.subheader("Model overview")
        st.write("ChurnIQ trains a Random Forest classifier on the supplied bank customer dataset. The target is explicitly selected as Exited; customer identifiers are excluded from training.")
        st.write("**Input variables**"); st.code(", ".join(numeric+categorical))
        st.markdown("**Risk bands**\\n- Low: below 30%\\n- Medium: 30% to below 70%\\n- High: 70% or above")
        st.info("Use predictions as decision support, not as the sole basis for customer treatment. Results on this sample dataset may not generalise to another institution or population.")
        st.write("Python · Pandas · Scikit-learn · Streamlit · Plotly")

if __name__=="__main__":
    main()
