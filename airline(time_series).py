#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt 
import statsmodels.api as sms
import pandas as pd


# In[4]:


df_airline=pd.read_csv('AirPassengers.csv')
df_airline.head()


# In[5]:


df_airline.isnull().sum()


# In[6]:


df_airline.tail()


# In[7]:


df_airline.info()


# In[11]:


# month column converting to datetime
df_airline['Month']=pd.to_datetime(df_airline['Month'])


# In[9]:


df_airline.info()


# In[12]:


df_airline.head()


# In[13]:


df_airline.set_index('Month',inplace=True)


# In[14]:


df_airline.head()


# In[15]:


df_airline.plot()


# In[16]:


from statsmodels.tsa.stattools import adfuller


# In[17]:


def adf_test(series):
    result=adfuller(series)
    print('ADF statistics: {}'.format(result[0]))
    print('p- value: {}'.format(result[1]))
    if result[1]<=0.5:
        print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print(print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary "))


# In[19]:


adf_test(df_airline['#Passengers'])


# In[20]:


#use tech differencing

df_airline['#Passengers first diffrence']=df_airline['#Passengers']-df_airline['#Passengers'].shift(1)


# In[21]:


df_airline.head()


# In[22]:


adf_test(df_airline['#Passengers first diffrence'].dropna())


# In[ ]:





# In[23]:


#use  tech diffrencecing
df_airline['Passanger second diff']=df_airline['#Passengers first diffrence']-df_airline['#Passengers first diffrence'].shift(1)


# In[24]:


adf_test(df_airline['Passanger second diff'].dropna())


# In[25]:


##season=12 months
# second time use tech diff

df_airline['Passanger 12 Differance']=df_airline['#Passengers']-df_airline['#Passengers'].shift(12)


# In[26]:


df_airline.head()


# In[27]:


adf_test(df_airline['Passanger 12 Differance'].dropna())


# In[28]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[29]:


acf=plot_acf(df_airline['Passanger second diff'].dropna())


# In[30]:


acf12=plot_acf(df_airline['Passanger 12 Differance'].dropna())


# In[32]:


pacf12=plot_pacf(df_airline['Passanger 12 Differance'].dropna())


# In[33]:


result=plot_pacf(df_airline['Passanger second diff'].dropna())


# In[34]:


#split train and test
df_airline


# In[35]:


from datetime import datetime,timedelta
train_dataset_end=datetime(1955,12,1)
test_dataset_end=datetime(1960,12,1)


# In[56]:


df_airline=df_airline.astype(float)


# In[57]:


train_data=df_airline[:train_dataset_end]
test_data=df_airline[train_dataset_end+timedelta(days=1):test_dataset_end]


# In[58]:


test_data


# In[59]:


#creating a ARIMA model

from statsmodels.tsa.arima_model import ARIMA


# In[60]:


model_arima=ARIMA(train_data['#Passengers'],order=(0,2,0))


# In[61]:


model_arima_fit=model_arima.fit()


# In[62]:


model_arima_fit.summary()


# In[63]:


#prediction

pred_start_Date=test_data.index[0]
pred_end_date=test_data.index[-1]
print(pred_start_Date)
print(pred_end_date)


# In[ ]:





# In[64]:


df_airline


# In[65]:


df_airline['#Passengers']


# In[66]:


pred=model_arima_fit.predict(start=pred_start_Date,end=pred_end_date)
residuals=test_data['#Passengers']-pred


# In[67]:


pred


# In[68]:


residuals


# In[69]:


model_arima_fit.resid.plot(kind='kde')


# In[70]:


test_data['preedicted arima']=pred


# In[72]:


test_data[['#Passengers','preedicted arima']].plot()


# In[73]:


## create a SARIMA model
from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[74]:


model_sarima=SARIMAX(test_data['#Passengers'],order=(3,0,5),seasonal_order=(0,1,0,12))


# In[76]:


model_SARIMA_fit=model_sarima.fit()


# In[77]:


model_SARIMA_fit.summary()


# In[78]:


##prediction
pred_start_date=test_data.index[0]
pred_end_date=test_data.index[-1]
print(pred_start_date)
print(pred_end_date)


# In[82]:


pred_Sarima=model_SARIMA_fit.predict(start=datetime(1956,1,1),end=datetime(1960,12,1))
residuals=test_data['#Passengers']-pred_Sarima


# In[83]:


model_SARIMA_fit.resid.plot()


# In[84]:


model_SARIMA_fit.resid.plot(kind='kde')


# In[85]:


test_data['Predicted_SARIMA']=pred_Sarima


# In[86]:


test_data[['#Passengers','Predicted_SARIMA','preedicted arima']].plot()


# In[ ]:





# In[ ]:




