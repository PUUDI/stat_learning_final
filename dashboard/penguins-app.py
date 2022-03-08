from operator import ge
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from datetime import date
 
#function for getting the difference between two dates
def numOfDays(date1, date2):
    return (date2-date1).days

#Define the functions that will be used in the app
def convert_dummy(df, feature,rank=0):
    pos = pd.get_dummies(df[feature], prefix=feature)
    mode = df[feature].value_counts().index[rank]
    biggest = feature + '_' + str(mode)
    pos.drop([biggest],axis=1,inplace=True)
    df.drop([feature],axis=1,inplace=True)
    df=df.join(pos)
    return df

def get_category(df, col, binsnum, labels, qcut = False):
    if qcut:
        localdf = pd.qcut(df[col], q = binsnum, labels = labels) # quantile cut
    else:
        localdf = pd.cut(df[col], bins = binsnum, labels = labels) # equal-length cut
        
    localdf = pd.DataFrame(localdf)
    name = 'gp' + '_' + col
    localdf[name] = localdf[col]
    df = df.join(localdf[name])
    df[name] = df[name].astype(object)
    return df

st.write("""
# Credit Card user Classification

This website will determine whether the new customer is eligible for a new Credit Card or not

Data obtained from [Kaggle Website](https://www.kaggle.com/sid321axn/eda-for-predicting-dementia/data)
""")

st.sidebar.header('Input the Patients Details')

st.sidebar.markdown("""
[Example CSV input file](https://gist.githubusercontent.com/PUUDI/861771ffca8462507b487b6f75f2386d/raw/44e4760f1f6ee628c9674fe1c87e63bd4fbcf19d/gistfile1.txt)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        
        gender = st.sidebar.selectbox('Gender',('Female','Male'))
        

        car = st.sidebar.selectbox('Own a car?',('Yes','No'))
        # car = car.replace(['No','Yes'],[0,1])

        realty = st.sidebar.selectbox('Own property?',('Yes','No'))
        # realty = car.replace(['No','Yes'],[0,1])

        phone = st.sidebar.selectbox('Own phone?',('Yes','No'))
        # phone = car.replace(['No','Yes'],[0,1])
        # phone = phone.astype(str)

        email = st.sidebar.selectbox('Has an email?',('Yes','No'))
        # email = car.replace(['No','Yes'],[0,1])
        # email = email.astype(str)

        wkphone = st.sidebar.selectbox('Have working phone number?',('Yes','No'))
        # wkphone = car.replace(['No','Yes'],[0,1])
        # wkphone = wkphone.astype(str)

        child_no = st.sidebar.slider('Number of children?', min_value =0,max_value =15,step =1)
        if (child_no >= 2):
            child_no ='2More'
        # input_df = convert_dummy(input_df,'ChldNo')

        inc = st.sidebar.number_input('Income',min_value= 0)
        # input_df['inc'] = input_df['inc']/10000
        # input_df = get_category(input_df,'inc', 3, ["low","medium", "high"], qcut = True)

        age = st.sidebar.number_input('Age', min_value =0,max_value =150,step =1)
        # input_df['Age']=-(input_df['DAYS_BIRTH'])//365
        # input_df = get_category(input_df,'Age',5, ["lowest","low","medium","high","highest"])
        # input_df = convert_dummy(input_df,'gp_Age')

        empDay = st.sidebar.date_input('Employement date')
        # input_df['worktm']=-(input_df['DAYS_EMPLOYED'])//365
        # input_df = get_category(input_df,'worktm',5, ["lowest","low","medium","high","highest"])
        # worktm = numOfDays(date.today(),empDay)
        # input_df = convert_dummy(input_df,'gp_worktm')

        famSize = st.sidebar.slider('Educational Rating', min_value = 0,max_value = 20,step = 1)
        # input_df.loc[input_df['famsizegp']>=3,'famsizegp']='3more'

        incomeType = st.sidebar.selectbox('Income type',('Working','Commercial associate','Pensioner','student','State servant'))
        # input_df.loc[input_df['inctp']=='Pensioner','inctp']='State servant'
        # input_df.loc[input_df['inctp']=='Student','inctp']='State servant'
        # input_df = convert_dummy(input_df,'inctp')

        occupation = st.sidebar.selectbox('Occupation',('Accountants','Cleaning staff','Cooking staff','Core staff','Drivers','High skill tech staff','HR staff','IT staff','Laborers','Low-skill Laborers','Managers','Medicine staff','Private service staff','Realty agents','Sales staff','Secretaries','Security staff','Waiters/barmen staff'))
        # input_df.loc[(input_df['occyp']=='Cleaning staff') | (input_df['occyp']=='Cooking staff') | (input_df['occyp']=='Drivers') | (input_df['occyp']=='Laborers') | (input_df['occyp']=='Low-skill Laborers') | (input_df['occyp']=='Security staff') | (input_df['occyp']=='Waiters/barmen staff'),'occyp']='Laborwk'
        # input_df.loc[(input_df['occyp']=='Accountants') | (input_df['occyp']=='Core staff') | (input_df['occyp']=='HR staff') | (input_df['occyp']=='Medicine staff') | (input_df['occyp']=='Private service staff') | (input_df['occyp']=='Realty agents') | (input_df['occyp']=='Sales staff') | (input_df['occyp']=='Secretaries'),'occyp']='officewk'
        # input_df.loc[(input_df['occyp']=='Managers') | (input_df['occyp']=='High skill tech staff') | (input_df['occyp']=='IT staff'),'occyp']='hightecwk'

        houseType = st.sidebar.selectbox('House Type',('Rented apartment','House / apartment','Co-op apartment','Municipal apartment','Office apartment','With parents'))
        # input_df = convert_dummy(input_df,'houtp')

        eduType = st.sidebar.selectbox('Education Type',('Academic degree','Higher Education','Incomplete higher','Lower secondary','Secondary / secondary special'))
        # input_df = convert_dummy(input_df,'edutp')

        famType = st.sidebar.selectbox('Family type',('Married','Single / not married','Civil marriage','Widow'))
        # input_df = convert_dummy(input_df,'famType')

        data = {'CODE_GENDER': gender,
                'FLAG_OWN_CAR': car,
                'FLAG_OWN_REALTY': realty,
                'FLAG_MOBIL':phone,
                'FLAG_EMAIL':email,
                'FLAG_WORK_PHONE':wkphone,
                'CNT_CHILDREN':child_no,
                'AMT_INCOME_TOTAL':inc,
                'DAYS_BIRTH':age,
                'DAYS_EMPLOYED':empDay,
                'CNT_FAM_MEMBERS':famSize,
                'NAME_INCOME_TYPE':incomeType,
                'OCCUPATION_TYPE':occupation,
                'NAME_HOUSING_TYPE':houseType,
                'NAME_EDUCATION_TYPE':eduType,
                'NAME_FAMILY_STATUS':famType}

        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()


# Pre processing the data to be predicted by the model(here we will try to make dataframe which is equal to the the oroginal DB)
input_df['Gender'] = input_df['Gender'].replace(['Female','Male'],[0,1])
input_df['Car'] = input_df['Car'].replace(['No','Yes'],[0,1])
input_df['Reality'] = input_df['Reality'].replace(['No','Yes'],[0,1])
input_df['phone']=input_df['phone'].astype(str)
input_df['email']=input_df['email'].astype(str)
input_df['wkphone']=input_df['wkphone'].astype(str)
input_df.loc[input_df['ChldNo'] >= 2,'ChldNo']='2More'
input_df = convert_dummy(input_df,'ChldNo')

input_df['inc']=input_df['inc'].astype(object)
input_df['inc'] = input_df['inc']/10000

input_df = get_category(input_df,'inc', 3, ["low","medium", "high"], qcut = True)
input_df = convert_dummy(input_df,'gp_inc')

input_df['Age']=-(input_df['DAYS_BIRTH'])//365

input_df = get_category(input_df,'Age',5, ["lowest","low","medium","high","highest"])

input_df = convert_dummy(input_df,'gp_Age')

input_df['worktm']=-(input_df['DAYS_EMPLOYED'])//365	
input_df[input_df['worktm']<0] = np.nan

input_df = get_category(input_df,'worktm',5, ["lowest","low","medium","high","highest"])

input_df = convert_dummy(input_df,'gp_worktm')

input_df['famsize']=input_df['famsize'].astype(int)
input_df['famsizegp']=input_df['famsize']
input_df['famsizegp']=input_df['famsizegp'].astype(object)
input_df.loc[input_df['famsizegp']>=3,'famsizegp']='3more'

input_df = convert_dummy(input_df,'famsizegp')

input_df.loc[input_df['inctp']=='Pensioner','inctp']='State servant'
input_df.loc[input_df['inctp']=='Student','inctp']='State servant'

input_df = convert_dummy(input_df,'inctp')

input_df = convert_dummy(input_df,'houtp')

input_df.loc[(input_df['occyp']=='Cleaning staff') | (input_df['occyp']=='Cooking staff') | (input_df['occyp']=='Drivers') | (input_df['occyp']=='Laborers') | (input_df['occyp']=='Low-skill Laborers') | (input_df['occyp']=='Security staff') | (input_df['occyp']=='Waiters/barmen staff'),'occyp']='Laborwk'
input_df.loc[(input_df['occyp']=='Accountants') | (input_df['occyp']=='Core staff') | (input_df['occyp']=='HR staff') | (input_df['occyp']=='Medicine staff') | (input_df['occyp']=='Private service staff') | (input_df['occyp']=='Realty agents') | (input_df['occyp']=='Sales staff') | (input_df['occyp']=='Secretaries'),'occyp']='officewk'
input_df.loc[(input_df['occyp']=='Managers') | (input_df['occyp']=='High skill tech staff') | (input_df['occyp']=='IT staff'),'occyp']='hightecwk'

input_df = convert_dummy(input_df,'occyp')

input_df.loc[input_df['edutp']=='Academic degree','edutp']='Higher education'

input_df = convert_dummy(input_df,'edutp')

input_df = convert_dummy(input_df,'famtp')




# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('NO CSV files are uploaded for prediction currently using the values entered manually')
    st.write(input_df)

# Reads in saved classification model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(input_df)
prediction_proba = load_clf.predict_proba(input_df)

#Creating the columns
col1,col2 = st.columns(2)

c1 = col1.container()
c2 = col2.container()


c1.subheader('Prediction')
penguins_species = np.array(['Negative','Positive'])
if prediction[0] == 0:
    status = 'The Patient is Negative'
else:
    status = 'The Patient is positive with Dementia'

c1.subheader(status)


c2.subheader('Prediction Probability')
c2.write(prediction_proba)

# go.Indicator(
#             mode="gauge+number+delta",
#             value=metrics["test_accuracy"],
#             title={"text": f"Accuracy (test)"},
#             domain={"x": [0, 1], "y": [0, 1]},
#             gauge={"axis": {"range": [0, 1]}},
#             delta={"reference": metrics["train_accuracy"]},
#         )
