import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics

data=pd.read_csv('studDS6.csv')
df=pd.DataFrame(data)
scaler=MinMaxScaler()

scaler.fit(df[['Age']])
df['Age']=scaler.transform(df[['Age']])
scaler.fit(df[['Height (in cms)']])
df['Height (in cms)']=scaler.transform(df[['Height (in cms)']])
scaler.fit(df[['Weight (in kgs)']])
df['Weight (in kgs)']=scaler.transform(df[['Weight (in kgs)']])
scaler.fit(df[['BMI']])
df['BMI']=scaler.transform(df[['BMI']])
scaler.fit(df[['Hours on Learning']])
df['Hours on Learning']=scaler.transform(df[['Hours on Learning']])
scaler.fit(df[['Hours on Extra-curricular']])
df['Hours on Extra-curricular']=scaler.transform(df[['Hours on Extra-curricular']])
scaler.fit(df[['Hours on Sleep Hygiene']])
df['Hours on Sleep Hygiene']=scaler.transform(df[['Hours on Sleep Hygiene']])


label_encoder = LabelEncoder()

df['GPA'] = label_encoder.fit_transform(df['GPA'])
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
pickle.dump(model, open("model.pkl", "wb"))