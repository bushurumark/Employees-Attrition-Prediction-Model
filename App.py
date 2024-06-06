#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import joblib


# In[2]:


# Load the trained model
model = joblib.load("model.pkl")


# In[3]:


# Define function to preprocess input data
def preprocess_input(Age, BusinessTravel, DailyRate, Department, DistanceFromHome,
                     Education, EducationField, EnvironmentSatisfaction, Gender,
                     HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction,
                     MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked,
                     OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction,
                     StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance,
                     YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrentManager):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Age': [Age],
        'BusinessTravel': [BusinessTravel],
        'DailyRate': [DailyRate],
        'Department': [Department],
        'DistanceFromHome': [DistanceFromHome],
        'Education': [Education],
        'EducationField': [EducationField],
        'EnvironmentSatisfaction': [EnvironmentSatisfaction],
        'Gender': [Gender],
        'HourlyRate': [HourlyRate],
        'JobInvolvement': [JobInvolvement],
        'JobLevel': [JobLevel],
        'JobRole': [JobRole],
        'JobSatisfaction': [JobSatisfaction],
        'MaritalStatus': [MaritalStatus],
        'MonthlyIncome': [MonthlyIncome],
        'MonthlyRate': [MonthlyRate],
        'NumCompaniesWorked': [NumCompaniesWorked],
        'OverTime': [OverTime],
        'PercentSalaryHike': [PercentSalaryHike],
        'PerformanceRating': [PerformanceRating],
        'RelationshipSatisfaction': [RelationshipSatisfaction],
        'StockOptionLevel': [StockOptionLevel],
        'TotalWorkingYears': [TotalWorkingYears],
        'TrainingTimesLastYear': [TrainingTimesLastYear],
        'WorkLifeBalance': [WorkLifeBalance],
        'YearsAtCompany': [YearsAtCompany],
        'YearsInCurrentRole': [YearsInCurrentRole],
        'YearsSinceLastPromotion': [YearsSinceLastPromotion],
        'YearsWithCurrentManager': [YearsWithCurrentManager]
    })

    # Encoding of all categorical variables 
    input_data_encoded = pd.get_dummies(input_data)
    
    # Ensure the input data matches the model's expected features
    model_features = model.feature_names_in_
    for feature in model_features:
        if feature not in input_data_encoded.columns:
            input_data_encoded[feature] = 0  # Add missing features with default value 0

    input_data_encoded = input_data_encoded[model_features]

    return input_data_encoded

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: blue;
    }
    .title {
        color: purple;
        text-align: center;
        font-size: 40px;
    }
    .widget-label {
        color: #ff6347;
        font-weight: bold;
    }
    .prediction-result {
        color: green;
        font-size: 30px;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Create the web interface
def main():
    st.markdown('<div class="title">Employees Attrition Prediction Model</div>', unsafe_allow_html=True)
    
    Age = st.number_input('Age', min_value=18, max_value=60)
    BusinessTravel = st.selectbox('BusinessTravel',  ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
    DailyRate = st.number_input('DailyRate', min_value=102, max_value=1500)
    Department = st.selectbox('Department', ['Research & Development', 'Sales', 'Human Resources'])
    DistanceFromHome = st.number_input('DistanceFromHome', min_value=1, max_value=29)
    Education = st.selectbox('Education', [1, 2, 3, 4, 5])
    EducationField = st.selectbox('EducationField', ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources'])
    EnvironmentSatisfaction = st.selectbox('EnvironmentSatisfaction', [1, 2, 3, 4])
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    HourlyRate = st.number_input('HourlyRate', min_value=30, max_value=100)
    JobInvolvement = st.selectbox('JobInvolvement', [1, 2, 3, 4])
    JobLevel = st.selectbox('JobLevel', [1, 2, 3, 4, 5])
    JobRole = st.selectbox('JobRole', ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
                                        'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director',
                                        'Human Resources'])
    JobSatisfaction = st.selectbox('JobSatisfaction', [1, 2, 3, 4])
    MaritalStatus = st.selectbox('MaritalStatus', ['Married', 'Single', 'Divorced'])
    MonthlyIncome = st.number_input('MonthlyIncome', min_value=1000, max_value=20000)
    MonthlyRate = st.number_input('MonthlyRate', min_value=2000, max_value=27000)
    NumCompaniesWorked = st.number_input('NumCompaniesWorked', min_value=0, max_value=10)
    OverTime = st.selectbox('OverTime', ['Yes', 'No'])
    PercentSalaryHike = st.number_input('PercentSalaryHike', min_value=11, max_value=25)
    PerformanceRating = st.selectbox('PerformanceRating', [1, 2, 3, 4])
    RelationshipSatisfaction = st.selectbox('RelationshipSatisfaction', [1, 2, 3, 4])
    StockOptionLevel = st.selectbox('StockOptionLevel', [0, 1, 2, 3])
    TotalWorkingYears = st.number_input('TotalWorkingYears', min_value=0, max_value=40)
    TrainingTimesLastYear = st.number_input('TrainingTimesLastYear', min_value=0, max_value=6)
    WorkLifeBalance = st.selectbox('WorkLifeBalance', [1, 2, 3, 4])
    YearsAtCompany = st.number_input('YearsAtCompany', min_value=0, max_value=40)
    YearsInCurrentRole = st.number_input('YearsInCurrentRole', min_value=0, max_value=18)
    YearsSinceLastPromotion = st.number_input('YearsSinceLastPromotion', min_value=0, max_value=15)
    YearsWithCurrentManager = st.number_input('YearsWithCurrManager', min_value=0, max_value=17)

    if st.button('Predict'):
        input_data = preprocess_input(Age, BusinessTravel, DailyRate, Department, DistanceFromHome,
                                      Education, EducationField, EnvironmentSatisfaction, Gender,
                                      HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction,
                                      MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked,
                                      OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction,
                                      StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance,
                                      YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrentManager)
        try:
            prediction = model.predict(input_data)[0]
            if prediction == 0:
                st.markdown('<div class="prediction-result">Prediction: Employee Will Not Attrite</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-result">Prediction: Employee Will Attrite</div>', unsafe_allow_html=True)
        except Exception as e:
            st.write(f"An error occurred: {e}")

if __name__ == '__main__':
    main()


# In[ ]:




