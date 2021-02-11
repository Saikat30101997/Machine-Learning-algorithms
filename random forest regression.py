import numpy as np
import pandas as  pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data=pd.read_csv(r"C:\Users\Saikat Das\AppData\Local\Programs\Python\Python38-32\petrol_consumption.csv")
print(data.head(10))

x=data.drop('Petrol_Consumption',axis=1).values
y=data['Petrol_Consumption'].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=20)

print(x_test)
sc=StandardScaler()

x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


from sklearn.ensemble import RandomForestRegressor

rge=RandomForestRegressor(n_estimators=50,random_state=0)
rge.fit(x_train,y_train)
prd=rge.predict(x_test)


from sklearn import metrics

mae=metrics.mean_absolute_error(y_test,prd)
print(mae)
mse=metrics.mean_squared_error(y_test,prd)
print(mse)
rmse=np.sqrt(mse)
print(rmse)

