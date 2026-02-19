import pandas as pd
import numpy as np

df = pd.read_csv('insurance - insurance.csv')
# print(df.head(3))

# print(df.isna().sum())

x = df.drop(columns=['charges'])
y = df['charges']

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

x['sex']=le.fit_transform(x['sex'])
x['smoker']=le.fit_transform(x['smoker'])
x['region']=le.fit_transform(x['region'])

# print(x.head(3))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

from sklearn.metrics import r2_score
print(r2_score(y_test , y_pred))

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, y_pred))

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred))