import numpy as np
import pandas as  pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data=pd.read_csv(r"C:\Users\Saikat Das\AppData\Local\Programs\Python\Python38-32\bill_authentication.csv")

print(data.head(10))

x=data.drop('Class',axis=1).values
y=data['Class'].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=50,random_state=20)
clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

cf=confusion_matrix(y_test,y_pred)
print(cf)
print(accuracy_score(y_test,y_pred))


