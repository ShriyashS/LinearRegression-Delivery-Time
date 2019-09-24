# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 10:33:51 2019

@author: Shriyash Shende
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as st

#Importing data set
dil = pd.read_csv('C:\\Users\\Good Guys\\Desktop\\pRACTICE\\EXCELR PYTHON\\Assignment\\Linear Regression\\delivery_time.csv')
#EDA
dil.describe()
dil.info()
dil.columns
np.median(dil['Delivery Time'])
np.median(dil['Sorting Time'])
sns.boxplot(dil['Delivery Time'])
sns.boxplot(x=dil['Sorting Time'])

sns.distplot(dil["Delivery Time"],bins=10,color="orange")
sns.distplot(dil['Sorting Time'],bins=10,color="brown")
sns.barplot(x=dil['Sorting Time'],y=dil["Delivery Time"],data=dil,orient="v")
#correlation value between X and Y
np.corrcoef(dil['Sorting Time'],dil['Delivery Time'])
plt.plot(dil['Sorting Time'],dil['Delivery Time'],"bo");plt.xlabel("Sorting Time");plt.ylabel("Delivery Time")
plt.show()

#Implementing Plot
model = st.ols("dil['Delivery Time'] ~ dil['Sorting Time']", data = dil).fit()
model.summary()
pred = model.predict(dil.iloc[:,-1])
plt.scatter(x=dil['Sorting Time'],y=dil['Delivery Time'],color='BLUE');plt.plot(dil['Sorting Time'],pred,color='red');plt.xlabel('SORTING TIME');plt.ylabel('DELIVERY TIME')
plt.show()
pred.corr(dil['Delivery Time'])

#TRANSFORMATION
#lOGRITHM TRANSFORMATION
model1 = st.ols("dil['Delivery Time'] ~ np.log(dil['Sorting Time'])", data = dil).fit()
model1.summary()
pred1 = model1.predict(pd.DataFrame(dil['Delivery Time']))
pred1.corr(dil['Delivery Time'])
plt.scatter(x=np.log(dil['Sorting Time']),y=dil['Delivery Time'],color='green');plt.plot(np.log(dil['Sorting Time']),pred1,color='blue');plt.xlabel('SORTING TIME');plt.ylabel('DELIVERY TIME')

#EXPONENTIAL TRANSFORMATION
model2 = st.ols("np.log(dil['Delivery Time']) ~ dil['Sorting Time']", data = dil).fit()
model2.summary()
pred2 = model2.predict(pd.DataFrame(dil['Delivery Time']))
pred_exp = np.exp(pred2)
pred_exp.corr(dil['Delivery Time'])
plt.scatter(x=dil['Sorting Time'],y=dil['Delivery Time'],color='green');plt.plot(dil['Sorting Time'],np.exp(pred2),color='blue');plt.xlabel('SORTING TIME');plt.ylabel('DELIVERY TIME')

## Quadratic model
dil['Sorting Time'] = dil['Sorting Time']* dil['Sorting Time']
model_quad = st.ols("dil['Delivery Time'] ~ dil['Sorting Time']",data=dil).fit()
model_quad.params
model_quad.summary()
pred_quad = model_quad.predict(pd.DataFrame(dil['Delivery Time']))
pred_quad.corr(dil['Delivery Time'])
model_quad.conf_int(0.05) # 
plt.scatter(dil['Sorting Time'],dil['Delivery Time'],c="b");plt.plot(dil['Sorting Time'],pred_quad,"r")
