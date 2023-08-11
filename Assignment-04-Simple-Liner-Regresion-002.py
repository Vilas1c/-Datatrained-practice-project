#!/usr/bin/env python
# coding: utf-8

# In[1]:


# impoort libraries
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[2]:


# import dataset
dataset=pd.read_csv('Salary_Data.csv')
dataset


# # performing EDA on data

# In[3]:


dataset.info()


# In[4]:


dataset.describe()
     


# # Checking for nul values

# In[5]:


dataset.isnull().sum()


# In[6]:


dataset[dataset.duplicated()].shape


# In[7]:


dataset[dataset.duplicated()]


# # Plotting the data to check for outliers

# In[8]:


plt.subplots(figsize=(9,6))
plt.subplot(121)
plt.boxplot(dataset['Salary'])
plt.title('Salary Hike')
plt.subplot(122)
plt.boxplot(dataset['YearsExperience'])
plt.title('years of experience')
plt.show()


# # Checking for correlation between variables

# In[9]:


dataset.corr()


# # Visualization of Correlation beteen x and y

# In[10]:


sns.regplot(x=dataset['YearsExperience'],y=dataset['Salary'])


# # Cheecking for homoscedacity and hetroscedacity

# In[11]:


plt.figure(figsize = (8,6), facecolor = 'lightgreen')
sns.scatterplot(x = dataset['YearsExperience'], y = dataset['Salary'])
plt.title('Homoscedasticity', fontweight = 'bold', fontsize = 16)
plt.show()


# In[12]:


dataset.var()


# # Feature Engineering

# In[13]:


sns.distplot(dataset['YearsExperience'], bins = 10, kde = True)
plt.title('Before Transformation')
sns.displot(np.log(dataset['YearsExperience']), bins = 10, kde = True)
plt.title('After Transformation')
plt.show()


# In[14]:


labels = ['Before Transformation','After Transformation']
sns.distplot(dataset['YearsExperience'], bins = 10, kde = True)
sns.distplot(np.log(dataset['YearsExperience']), bins = 10, kde = True)
plt.legend(labels)
plt.show()


# In[15]:


sm.qqplot(np.log(dataset['YearsExperience']), line = 'r')
plt.title('No transformation')
sm.qqplot(np.sqrt(dataset['YearsExperience']), line = 'r')
plt.title('Log transformation')
sm.qqplot(np.sqrt(dataset['YearsExperience']), line = 'r')
plt.title('Square root transformation')
sm.qqplot(np.cbrt(dataset['YearsExperience']), line = 'r')
plt.title('Cube root transformation')
plt.show()


# In[16]:


labels = ['Before Transformation','After Transformation']
sns.distplot(dataset['Salary'], bins = 10, kde = True)
sns.displot(np.log(dataset['Salary']), bins = 10, kde = True)
plt.title('After Transformation')
plt.show()


# In[17]:


sm.qqplot(dataset['Salary'], line = 'r')
plt.title('No transformation')
sm.qqplot(np.log(dataset['Salary']), line = 'r')
plt.title('Log transformation')
sm.qqplot(np.sqrt(dataset['Salary']), line = 'r')
plt.title('Square root transformation')
sm.qqplot(np.cbrt(dataset['Salary']), line = 'r')
plt.title('Cube root transformation')
plt.show()


# # fitting a linear regression model
# 
# Using Ordinary least squares (OLS) regression

# In[18]:


model = smf.ols('Salary~YearsExperience',data=dataset).fit()
     

model.summary()


# # Square Root transformation on data

# In[19]:


model1=smf.ols('np.sqrt(Salary)~np.sqrt(YearsExperience)',data=dataset).fit()
model1.summary()


# # Cuberoot transformation on Data

# In[20]:


model2=smf.ols('np.cbrt(Salary)~np.cbrt(YearsExperience)',data=dataset).fit()
     

model2.summary()


# # Log transformation on Data

# In[21]:


model3=smf.ols('np.log(Salary)~np.log(YearsExperience)',data=dataset).fit()
model3.summary()


# # Model Testing
# 
# As Y = Beta0 + Beta1*(X)
# 
# Finding Coefficient Parameters (Beta0 and Beta1 values)

# In[22]:


model.params


# In[23]:


print(model.tvalues,'\n',model.pvalues)


# # Residual Analysis
# 
# Test for Normality of Residuals (Q-Q Plot)

# In[24]:


sm.qqplot(model.resid,line='q')
plt.title('normal qqplot of model without any data transformation')
plt.show()


# # Residual Plot to check Homoscedasticity or Hetroscedasticity

# In[25]:


def get_standardized_values( vals ):
  return (vals - vals.mean())/vals.std()


# In[26]:


plt.scatter(get_standardized_values(model.fittedvalues),get_standardized_values(model.resid))
plt.title('residual plot for model without any data transformation')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show()


# # Model Validation

# In[27]:


from sklearn.metrics import mean_squared_error


# In[28]:


model1_pred_y =np.square(model1.predict(dataset['YearsExperience']))
model2_pred_y =pow(model2.predict(dataset['YearsExperience']),3)
model3_pred_y =np.exp(model3.predict(dataset['YearsExperience']))
     

model1_rmse =np.sqrt(mean_squared_error(dataset['Salary'], model1_pred_y))
model2_rmse =np.sqrt(mean_squared_error(dataset['Salary'], model2_pred_y))
model3_rmse =np.sqrt(mean_squared_error(dataset['Salary'], model3_pred_y))
print('model=', np.sqrt(model.mse_resid),'\n' 'model1=', model1_rmse,'\n' 'model2=', model2_rmse,'\n' 'model3=', model3_rmse)
     


# In[29]:


rmse = {'model': np.sqrt(model.mse_resid), 'model1': model1_rmse, 'model2': model3_rmse, 'model3' : model3_rmse}
min(rmse, key=rmse.get)


# In[ ]:


#As model has the minimum RMSE and highest Adjusted R-squared score. Hence, we are going to use model to predict our values

#Model is that Simple Linear regression model where we did not perfrom any data transformation and got the highest Adjusted R-squared value


# # predicting values

# In[30]:


# first model results without any transformation
predicted2 = pd.DataFrame()
predicted2['YearsExperience'] = dataset.YearsExperience
predicted2['Salary'] = dataset.Salary
predicted2['Predicted_Salary_Hike'] = pd.DataFrame(model.predict(predicted2.YearsExperience))
predicted2
     


# In[ ]:





# In[ ]:




