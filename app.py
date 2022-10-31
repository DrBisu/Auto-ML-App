import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import sklearn
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFECV

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Auto Med ML")
    choice = st.radio("Navigation", ["Data Upload", "Data Profile", "Data Pre-processing", "ML Modelling", "Download model"])
    st.info("An automated machine learning tool for medical research. Created by medical students for medical students, with the goal of making machine learning accessible to all.")
    st.info("Created by: AI Lab in Neurosurgery, Manchester Centre for Clinical Neurosciences, Salford Royal Hospital, Manchester, UK")

if choice == "Data Upload": 
    st.title("Data Upload")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_excel(file, index_col=None)
        df.to_csv("dataset.csv", index=None)
        st.dataframe(df)
        
if os.path.exists("dataset.csv"): 
    df = pd.read_csv('dataset.csv', index_col=None)

if choice == "Data Profile": 
    st.title("Data Profile")
    profile_df = df.profile_report()
    st_profile_report(profile_df)
    
if choice == "Data Pre-processing":
    target = st.selectbox("Choose the Target", df.columns)
    n_splits = st.slider('Number of CV Splits', 2, 15)
    n_repeats = st.slider('Number of CV Repeats', 1, 15)
    KFolds = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
    X = df.drop(target, axis=1)
    Y = df[target]
    
    
    #Code for Recursive Feature Elimination
    model = LogisticRegression(solver='lbfgs', max_iter=1000, penalty="l2")
    min_features_to_select = 1  # Minimum number of features to consider
    rfecv = RFECV(
        estimator=model,
        step=1,
        cv=KFolds,
        scoring="roc_auc",
        min_features_to_select=min_features_to_select,
    )
    rfecv.fit(X, Y)
    imp_features = rfecv.get_feature_names_out()
    imp_features = pd.DataFrame(imp_features)
    st.dataframe(imp_features)
    
    #Code for dropping variables
    
if choice == "ML Modelling":
        classifier_name = st.selectbox("Choose the Classifier", ["Logistic Regression", "KNN", "SVM", "Decision Tree", "Random Forest"])
        st.button("Run Modelling")
    
if choice == "Download model":
    with open("best_model.pkl", 'rb') as f: 
        st.download_button("Download Model", f, "best_model_test.pkl")