#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as smf
import statsmodels.formula.api as sm
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# import dataset
dataset=pd.read_csv('delivery_time.csv')
dataset


# # Performing EDA on Data

# In[3]:


# renaming columns
dataset1 = dataset.rename({'Delivery Time':'delivery_time','Sorting Time':'sorting_time'},axis=1)
dataset1


# In[4]:


# checking datatype
dataset.info()


# In[5]:


dataset.describe()


# In[6]:


# checking for null values
dataset.isnull().sum()


# In[7]:


# checking for duplicate values
dataset[dataset.duplicated()].shape


# In[8]:


dataset[dataset.duplicated()]


# # Plotting the data to check for outliers

# In[9]:


plt.subplots(figsize = (9,6))
plt.subplot(121)
plt.boxplot(dataset['Delivery Time'])
plt.title('Delivery Time')
plt.subplot(122)
plt.boxplot(dataset['Sorting Time'])
plt.title('Sorting Time')
plt.show()


# In[10]:


# checking the correlation 
dataset.corr()


# # Visualization of correlation between x and y

# In[11]:


# regression plot
sns.regplot(x=dataset1['sorting_time'],y=dataset1['delivery_time'])


# # Checking for Homoscedasticity or Hetroscedasticity

# In[12]:


plt.figure(figsize=(8,6))
sns.scatterplot(x=dataset1['sorting_time'],y=dataset1['delivery_time'])
plt.title('Hetroscedasticity',fontsize=18)
plt.show()


# In[13]:


dataset.var()


# # Feature engineering

# In[14]:


sns.distplot(dataset['Delivery Time'],bins=10,kde=True)
plt.title('Before Transformation')
sns.displot(np.log(dataset['Delivery Time']),bins=10,kde=True)
plt.title('After Transformation')
plt.show()


# In[15]:


labels = ['Before Transformation','After Transformation']
sns.distplot(dataset['Delivery Time'],bins=10,kde=True)
sns.distplot(np.log(dataset['Delivery Time']),bins=10,kde=True)
plt.legend(labels)
plt.show()


# In[16]:


smf.qqplot(dataset['Delivery Time'], line = 'r')
plt.title('No transformation')
smf.qqplot(np.log(dataset['Delivery Time']),line='r')
plt.title('log transformation')
smf.qqplot(np.sqrt(dataset['Delivery Time']),line='r')
plt.title('square root transformation')
smf.qqplot(np.cbrt(dataset['Delivery Time']),line='r')
plt.title('cube root transformation')
plt.show()


# In[17]:


labels=['Before Transformation','After Transformation']
sns.distplot(dataset['Sorting Time'],bins=10,kde=True)
sns.distplot(np.log(dataset['Sorting Time']), bins = 10, kde = True)
plt.legend(labels)
plt.show()
     


# In[18]:


smf.qqplot(dataset['Sorting Time'],line='r')
plt.title('No transformation')
smf.qqplot(np.log(dataset['Sorting Time']),line='r')
plt.title('Log Transformation')
smf.qqplot(np.sqrt(dataset['Sorting Time']),line='r')
plt.title('square root transformation')
smf.qqplot(np.cbrt(dataset['Sorting Time']),line='r')
plt.title('cube root transformation')
plt.show()


# # Fitting a Linear Regression Model
# 
# 

# In[19]:


model = sm.ols('delivery_time~sorting_time',data=dataset1).fit()
     

model.summary()


# # Square Root transformation on data

# In[20]:


model1=sm.ols('np.sqrt(delivery_time)~np.sqrt(sorting_time)',data=dataset1).fit()
model1.summary()
     


# # Cube Root transformation on Data

# In[21]:


model2=sm.ols('np.cbrt(delivery_time)~np.cbrt(sorting_time)',data=dataset1).fit()
model2.summary()
     


# # Log transformation on Data

# In[22]:


model3=sm.ols('np.log(delivery_time)~np.log(sorting_time)',data=dataset1).fit()
model3.summary()


# # Model Testing
# 
# As Y = Beta0 + Beta1*(X)
# 
# Finding Coefficient Parameters (Beta0 and Beta1 values)

# In[23]:


model.params


# In[24]:


print(model.tvalues,'\n',model.pvalues)


# In[25]:


model.rsquared,model.rsquared_adj


# # Residual analysis
# 
# Test for Normality of Residuals (qq plot)

# In[26]:


import statsmodels.api as sm
sm.qqplot(model.resid, line='q')
plt.title('Normal Q-Q plot of residuals of model without any data transformation')
plt.show()
     


# In[27]:


sm.qqplot(model2.resid, line='q')
plt.title('Normal Q-Q plot of residuals of model with log transformation')
plt.show()


# # Residual plot to check Homoscedacity or hetroscedacity
# 
# 

# In[28]:


def get_standardized_values(vals):
  return (vals-vals.mean())/vals.std()


# In[29]:


plt.scatter(get_standardized_values(model.fittedvalues),get_standardized_values(model.resid))
plt.title('Residual plot for model without any data transformation')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show()
     


# In[30]:


plt.scatter(get_standardized_values(model2.fittedvalues), get_standardized_values(model2.resid))
plt.title('Residual Plot for Model with Log transformation')
plt.xlabel('Standardized Features Values')
plt.ylabel('Standardized Residual Values')
plt.show()


# # Model validation

# In[31]:


from sklearn.metrics import mean_squared_error


# In[32]:


model1_pred_y=np.square(model1.predict(dataset1['sorting_time']))
model2_pred_y=pow(model2.predict(dataset1['sorting_time']),3)
model3_pred_y=np.exp(model3.predict(dataset1['sorting_time']))


# In[33]:


model1_rmse=np.sqrt(mean_squared_error(dataset1['delivery_time'],model1_pred_y))
model2_rmse=np.sqrt(mean_squared_error(dataset1['delivery_time'],model2_pred_y))
model3_rmse=np.sqrt(mean_squared_error(dataset1['delivery_time'],model3_pred_y))
print('model1=',np.sqrt(model.mse_resid),'\n' 'model1=',model1_rmse,'\n' 'model2=',model2_rmse, '\n' 'model3=',model3_rmse)
     


# In[34]:


data={'model':np.sqrt(model.mse_resid),'model1':model1_rmse,'model2':model3_rmse,'model3':model3_rmse}
min(data,key=data.get)

model2 has the minimum RMSE and highest Adjusted R-squared score. Hence, we are going to use model2 to predict our values
# # predicting the values from model with log transformation on the data

# In[35]:


predicted=pd.DataFrame()
predicted['sorting_time']=dataset1.sorting_time
predicted['delivery_time']=dataset1.delivery_time
predicted['predicted_delivery_time']=pd.DataFrame(np.exp(model2.predict(predicted.sorting_time)))
predicted


# In[36]:


predicted1=pd.DataFrame()
predicted1['sorting_time']=dataset1.sorting_time
predicted1['delivery_time']=dataset1.delivery_time
predicted1['predicted_delivery_time']=pd.DataFrame(np.exp(model3.predict(predicted1.sorting_time)))
predicted1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




