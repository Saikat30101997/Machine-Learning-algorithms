import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\Saikat Das\AppData\Local\Programs\Python\Python38-32\headbrain.csv")
print(data.shape)

x = data['Head Size(cm^3)'].values
y=data['Brain Weight(grams)'].values


mean_x = np.mean(x)
mean_y=np.mean(y)

print(mean_x,mean_y)

n = len(x)
lob =0
hor = 0
i=0
while i<n:
    lob+=(x[i]-mean_x)*(y[i]-mean_y)
    hor+=(x[i]-mean_x)**2
    i+=1
m=lob/hor
c = mean_y-(m*mean_x)
print(m,c)

y1=[]
i=0
while i<n:
    t=x[i]*m+c
    y1.append(t)
    i+=1
y1=np.array(y1)

lob1=0
hor1=0

i=0
while i<n:
    lob1+=(y1[i]-mean_y)**2
    hor1+=(y[i]-mean_y)**2
    i+=1
r_square=1-(lob1/hor1)
print(r_square)

plt.scatter(x,y,color='g')
plt.plot(x,y1,color='r')
plt.show()

