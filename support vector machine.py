import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.svm import SVC
import seaborn as sns

data = pd.read_csv(r"C:\Users\Saikat Das\AppData\Local\Programs\Python\Python38-32\datasets_19_420_Iris.csv")
data.drop('Id',axis=1,inplace=True)
x=data[data.Species=='Iris-setosa']
y=data[data.Species=='Iris-versicolor']
plt.scatter(x['SepalLengthCm'],x['SepalWidthCm'],color='b',marker='+')
plt.scatter(y['SepalLengthCm'],y['SepalWidthCm'],color='g',marker='.')
plt.show()

x1=data.drop('Species',axis=1).values
y1=data['Species'].values
print(x1)
print(y1)
x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.25,random_state=60)
model = SVC(C=50,kernel='linear',gamma='scale')
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
y_pred=model.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
cm_df=pd.DataFrame(cm,index=['S','ver','vir'],columns=['S','ver','vir'])
sns.heatmap(cm_df,annot=True)
plt.show()




