import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import pandas as pd

acc_data=pd.read_csv(r"C:\Users\Saikat Das\AppData\Local\Programs\Python\Python38-32\accent-mfcc-data-1.csv")
print(acc_data.head(10))


ac = acc_data.iloc[:, 1:13]
ac = pd.DataFrame(ac)
ac.plot.kde()
from sklearn.preprocessing import RobustScaler
r=RobustScaler()

x=acc_data.drop("language",axis=1)
y=acc_data["language"]
new_x=r.fit_transform(x)
print(new_x)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
new_y=le.fit_transform(y)
print(le.classes_)

x_train,x_test,y_train,y_test=train_test_split(new_x,new_y,test_size=0.25,random_state=31)
x_train=r.fit_transform(x_train)
x_test=r.transform(x_test)
gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)
acc=accuracy_score(y_test,y_pred)
print(acc)

cf=confusion_matrix(y_test,y_pred)
print(cf)