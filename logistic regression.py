from sklearn.model_selection import train_test_split ## train_test_split package amder algo te data train r test duita vage vag kore nite pari er maddhome amra koto percent train korbo data r koto prcnt testkorbo ta detect korte pari
from sklearn.linear_model import LogisticRegression ## logisitic regression model import korlam
from sklearn.metrics import classification_report##A Classification report is used to measure the quality of predictions from a classification algorithm. How many predictions are True and how many are False.
from sklearn.metrics import confusion_matrix ## er maddhome amra accuracy ber korte parboo tai import bar korlam
from sklearn.metrics import accuracy_score## accuracy score bar korar jonno
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

##collecting data
titanic_data = pd.read_csv(r"C:\Users\Saikat Das\AppData\Local\Programs\Python\Python38-32\titanic.csv")##csv read
##p = sns.load_dataset(r'C:\Users\Saikat Das\AppData\Local\Programs\Python\Python38-32\titanic')

##print(titanic_data)
##print((len(titanic_data)))

##analysing data
sns.countplot(x="Survived",data=titanic_data)
plt.show()

sns.countplot(x="Survived",hue="Pclass",data=titanic_data)##count graph dekhabe survived koyjon
plt.show()

titanic_data["Age"].plot.hist()
plt.show()

titanic_data["Fare"].plot.hist(bins=20,figsize=(10,5))## fare coloumn hist graph e show krbe
plt.show()



sns.countplot(x="SibSp",data=titanic_data)
plt.show()

#data wrangling

print(titanic_data.isnull().sum())## koyjon nan column e ase show korbe


sns.heatmap(titanic_data.isnull(),yticklabels=False)
plt.show()

sns.boxplot(x="Pclass",y="Age",data=titanic_data)
plt.show()

titanic_data.drop("Cabin",axis=1,inplace=True)##cabin er data drop korbe mane cabin er column delete inplace na dike hobe na
titanic_data.dropna(inplace=True) ## onnano coloumn e jegula nan delete kore dibe inplace na dile hobe naa

sns.heatmap(titanic_data.isnull(),yticklabels=False)##heatmap diye dekhabe graph j koyjon nan ase..
plt.show()

print(titanic_data.isnull().sum())

##convert string into categorial variable

sex=pd.get_dummies(titanic_data['Sex'],drop_first=True)##drop korche string gulaa
print(sex.head(10))##10 sex column cat.var print koraaa

embark=pd.get_dummies(titanic_data['Embarked'],drop_first=True)##embarked cat,var e print koraa..
print(embark.head(10))

##amra pclass kew o categorial e ante hobe

pclass=pd.get_dummies(titanic_data['Pclass'],drop_first=True)
print(pclass.head(5))

##amder ja categorial korlam ta actual dataset e hy nai amra akhon eta k pandas er concatenet fnc er maddhome add korboo

titanic_data=pd.concat([titanic_data,sex,embark,pclass],axis=1)##variable declare kora + concat fnc 1st third bracket 1st variable jeta csv deeclare e nlam then amra jegula convert korsi oigula add kroboo and axis =1 dbo jatey clmn wise change hy
print(titanic_data.head(10))

##drop string coulmn

titanic_data.drop(['Sex','Pclass','PassengerId','Name','Ticket','Embarked'],axis=1,inplace=True) ## delete kora inplace na use korle delete hobe na
print(titanic_data.head(10))

##Train & test data

x=titanic_data.drop('Survived',axis=1) ## ekhane survived coloumn badey shob gula e independent
y=titanic_data['Survived'] ## amra survived k dependent nboo
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=20)## ekhane 4 ta variable declare korlam then amra fncnt er maddhome x r y variable bosalam test_size diye amra 0.2 mane 20 % test korbe r 80% train korbe r random diye holo 20 variable bar bar random nibe ek rokom nibe na
logmodel=LogisticRegression() ##declare korlam variable e
logmodel.fit(X_train,Y_train)## fnctn ta te amra train variable declare koree dilaam  fit fnctn diye declare korte hoy

##accuracy_check
predictions=logmodel.predict(X_test) ## test er data amra predict fncnt er maddhome nilaam

print(classification_report(Y_test,predictions))

m=confusion_matrix(Y_test,predictions)## confusion matrix are 2by 2 matrix to get hw the accurate the values coloumn predictive no predictive yes r row te thakbe actual no r actual yes
print(m)

## print korale 105 r 61 holo jothakrome (PN,AN) and (PY,AY) eder jogfoll re mot sum dara vag korle ans pawoaa jabe accuracy
n=accuracy_score(Y_test,predictions)## ekhane dependent test r prediction value bosabo...
print(n) ## 78% dekhabe accuracy (163/214)



