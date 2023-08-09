#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# In[3]:


salary=pd.read_csv("SalaryData_Test.csv")


# In[4]:


salary


# In[5]:


salary.head()


# In[6]:


salary.info()


# In[7]:


salary['workclass']=salary['workclass'].astype('category')
salary['education']=salary['education'].astype('category')
salary['maritalstatus']=salary['maritalstatus'].astype('category')
salary['occupation']=salary['occupation'].astype('category')
salary['relationship']=salary['relationship'].astype('category')
salary['race']=salary['race'].astype('category')
salary['native']=salary['native'].astype('category')
salary['sex']=salary['sex'].astype('category')


# In[8]:


salary.dtypes


# In[9]:


from sklearn import preprocessing                      
label_encoder = preprocessing.LabelEncoder()


# In[10]:


salary['Salary'] = label_encoder.fit_transform(salary['Salary'])


# In[12]:


salary.Salary


# In[13]:


#we also need to convert categories into numbers
salary['workclass'] = label_encoder.fit_transform(salary['workclass'])
salary['education'] = label_encoder.fit_transform(salary['education'])
salary['maritalstatus'] = label_encoder.fit_transform(salary['maritalstatus'])
salary['occupation'] = label_encoder.fit_transform(salary['occupation'])
salary['relationship'] = label_encoder.fit_transform(salary['relationship'])
salary['race'] = label_encoder.fit_transform(salary['race'])
salary['sex'] = label_encoder.fit_transform(salary['sex'])
salary['native'] = label_encoder.fit_transform(salary['native'])


# In[14]:


salary


# In[15]:


#define x&y

X = salary.iloc[:,0:13]
Y = salary.iloc[:,13]


# In[17]:


X


# In[18]:


Y


# In[19]:


salary.Salary.unique()


# In[20]:


salary.Salary.value_counts()


# In[21]:


# Splitting the data into training and test dataset

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=0)


# In[22]:


#model building by using SVM
clf=SVC()
clf.fit(x_train , y_train)
y_pred = clf.predict(x_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# In[23]:


y_pred=clf.predict(x_test)
y_pred


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




