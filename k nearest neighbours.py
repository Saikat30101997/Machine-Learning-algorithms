import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns


data=pd.read_csv(r'C:\Users\Saikat Das\AppData\Local\Programs\Python\Python38-32\datasets_19_420_Iris.csv')

x=data.iloc[:,1:5].values
y=data['Species'].values


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=20,test_size=0.2)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

cls=KNeighborsClassifier(n_neighbors=5,algorithm='brute')
cls.fit(x_train,y_train)
y_pred=cls.predict(x_test)

acc=accuracy_score(y_pred,y_test)
print(acc)
cl=classification_report(y_pred,y_test)
print(cl)

cm=confusion_matrix(y_pred,y_test)
print(cm)

cm_df=pd.DataFrame(cm,index=['S','ver','vir'],columns=['S','ver','vir'])
sns.heatmap(cm_df,annot=True)
plt.title(f'Accuracy_score {accuracy_score(y_pred,y_test)}')
plt.show()

