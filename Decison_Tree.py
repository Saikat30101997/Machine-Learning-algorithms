import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv(r'C:\Users\Saikat Das\AppData\Local\Programs\Python\Python38-32\dataset_11_balance-scale.csv')

print(data.head(6))

'''cl=pd.get_dummies(data['class'],drop_first=True)
print(cl.head(5))
data.drop(['class'],axis=1,inplace=True)
data=data.concat([data,cl],'''
print(data.head(6))
x=data.drop('class',axis=1).values
y=data['class'].values

print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)

clf_gini=DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=3,min_samples_leaf=5)

clf_gini.fit(x_train,y_train)

clf_entropy=DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=3,min_samples_leaf=5)

clf_entropy.fit(x_train,y_train)

y_pred_gini=clf_gini.predict(x_test)
y_pred_entropy=clf_entropy.predict(x_test)

y_gini_confusion_matrix=confusion_matrix(y_test,y_pred_gini)
print(y_gini_confusion_matrix)

y_gini_acc=accuracy_score(y_test,y_pred_gini)
print(y_gini_acc*100)

print(classification_report(y_test,y_pred_gini))

y_entropy_c_m=confusion_matrix(y_test,y_pred_entropy)
print(y_entropy_c_m)

y_en_acc=accuracy_score(y_test,y_pred_entropy)

print(y_en_acc*100)

print(classification_report(y_test,y_pred_entropy))
