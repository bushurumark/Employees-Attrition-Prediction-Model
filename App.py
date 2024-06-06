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
def preprocess_input(Age, BusinessTravel,DailyRate,Department,DistanceFromHome,
                     Education,EducationField,EnvironmentSatisfaction,Gender,
                     HourlyRate,JobInvolvement,JobLevel,JobRole,JobSatisfaction,
                     MaritalStatus,MonthlyIncome,MonthlyRate,NumCompaniesWorked,
                     OverTime,PercentSalaryHike,PerformanceRating,RelationshipSatisfaction,
                     StockOptionLevel,TotalWorkingYears,TrainingTimesLastYear,WorkLifeBalance,
                     YearsAtCompany,YearsInCurrentRole,YearsSinceLastPromotion,YearsWithCurrManager):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Age': [Age],
        'Business_Travel': [BusinessTravel],
        'Daily_Rate': [DailyRate],  
        'Department': [Department],
        'Distance_From_Home': [DistanceFromHome],
        'Education': [Education],
        'Education_Field': [EducationField],
        'Environment_Satisfaction': [EnvironmentSatisfaction],
        'Gender': [Gender],
        'Hourly_Rate': [HourlyRate],
        'Job_Involvement': [JobInvolvement],
        'Job_Level': [JobLevel],
        'Job_Role': [JobRole],
        'Job_Satisfaction': [JobSatisfaction],
        'Marital_Status': [MaritalStatus],
        'Monthly_Income': [MonthlyIncome],
        'Monthly_Rate': [MonthlyRate],
        'Numbers_of_Companies_Worked': [NumCompaniesWorked],
        'Over_Time': [OverTime],
        'Percentage_Salary_Hike': [PercentSalaryHike],
        'Performance_Rating': [PerformanceRating],
        'Relationship_Satisfaction': [RelationshipSatisfaction],
        'Stock_Option_Level': [StockOptionLevel],
        'Total_Working_Years': [TotalWorkingYears],
        'Training_Times_Last_Year': [TrainingTimesLastYear],
        'Work_Life_Balance': [WorkLifeBalance],
        'Years_At_Company': [YearsAtCompany],
        'Years_In_Current_Role': [YearsInCurrentRole],
        'Years_Since_Last_Promotion': [YearsSinceLastPromotion],
        'Years_With_Current_Manager': [YearsWithCurrManager]
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
    Business_Travel = st.selectbox('BusinessTravel',  ['Travel_Rarely','Travel_Frequently','Non-Travel'])
    Daily_Rate = st.number_input('DailyRate', min_value=102, max_value=1500)
    Department = st.selectbox('Department', ['Research & Development','Sales','Human Resources'])
    Distance_From_Home = st.number_input('DistanceFromHome', min_value=1, max_value=29)
    Education = st.selectbox('Education', [1,2,3,4,5])
    EducationField = st.selectbox('EducationField', ['Life Sciences','Medical','Marketing','Technical Degree','Other','Human Resources'])
    Environment_Satisfaction = st.selectbox('EnvironmentSatisfaction', [1,2,3,4])
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    HourlyRate = st.number_input('HourlyRate', min_value=30, max_value=100)
    Job_Involvement = st.selectbox('JobInvolvement', [1,2,3,4])
    Job_Level = st.selectbox('JobLevel', [1,2,3,4,5])
    Job_Role = st.selectbox('JobRole', ['Sales Executive','Research Scientist','Laboratory Technician','Manufacturing Director',
                                        'Healthcare Representative','Manager','Sales Representative','Research Director',
                                        'Human Resources'])
    Job_Satisfaction = st.selectbox('JobSatisfaction', [1,2,3,4])
    Marital_Status = st.selectbox('MaritalStatus', ['Married','Single','Divorced'])
    Monthly_Income = st.number_input('MonthlyIncome', min_value=1000, max_value=20000)
    Monthly_Rate = st.number_input('MonthlyRate', min_value=2000, max_value=27000)
    Number_of_Companies_Worked = st.number_input('NumCompaniesWorked', min_value=0, max_value=10)
    Over_Time = st.selectbox('OverTime', ['Yes','No'])
    Percentage_Salary_Hike = st.number_input('PercentSalaryHike', min_value=11, max_value=25)
    Performance_Rating = st.selectbox('PerformanceRating', [1,2,3,4])
    Relationship_Satisfaction = st.selectbox('RelationshipSatisfaction', [1,2,3,4])
    Stock_Option_Level = st.selectbox('StockOptionLevel',[0,1,2,3])
    Total_Working_Years = st.number_input('TotalWorkingYears', min_value=0, max_value=40)
    Training_Times_Last_Year = st.number_input('TrainingTimesLastYear', min_value=0, max_value=6)
    Work_Life_Balance = st.selectbox('WorkLifeBalance', [1,2,3,4])
    Years_At_Company = st.number_input('YearsAtCompany', min_value=0, max_value=40)
    Years_In_Current_Role = st.number_input('YearsInCurrentRole', min_value=0, max_value=18)
    Years_Since_Last_Promotion = st.number_input('YearsSinceLastPromotion', min_value=0, max_value=15)
    Years_With_Current_Manager = st.number_input('YearsWithCurrManager', min_value=0, max_value=17)

    if st.button('Predict'):
        input_data = preprocess_input(Age, BusinessTravel,DailyRate,Department,DistanceFromHome,
                                      Education,EducationField,EnvironmentSatisfaction,Gender,
                                      HourlyRate,JobInvolvement,JobLevel,JobRole,JobSatisfaction,
                                      MaritalStatus,MonthlyIncome,MonthlyRate,NumCompaniesWorked,
                                      OverTime,PercentSalaryHike,PerformanceRating,RelationshipSatisfaction,
                                      StockOptionLevel,TotalWorkingYears,TrainingTimesLastYear,WorkLifeBalance,
                                      YearsAtCompany,YearsInCurrentRole,YearsSinceLastPromotion,YearsWithCurrManager)
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




