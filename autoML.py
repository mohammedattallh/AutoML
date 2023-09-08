
import streamlit as st
import plotly.express as px
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import pandas as pd
import os

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload","EDA","Modelling","Evaluation" ,"Download"])
    st.info("This project application helps you build and explore your data.")


if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
if choice == 'EDA':
     st.title("Exploratory Data Analysis")
     profile_df = df.profile_report()
     st_profile_report(profile_df)  
        

if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if df[chosen_target].dtype == 'int64' or 'float64':
       from pycaret.regression import *
    elif df[chosen_target].dtype == 'object' or 'bool':
       from pycaret.classification import *
    train_size = st.slider('determine train set size',1,100) 
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target,train_size = train_size*0.01)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        st.info('choose the best model for you')
        chosen_model = st.text_input('Enter name model from column abbreviations ')
        btn = st.button('create model')
        if btn:
            cremod = create_model(chosen_model)
            cremod_2 = pull()
            st.dataframe(cremod_2)

if choice == "Evaluation":
    tunmod = tune_model(cremod,choose_better = True )
    evaluate_model(tunmod)
    save_model(tunmod,'best_model')
    

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="trainee_model.pkl")