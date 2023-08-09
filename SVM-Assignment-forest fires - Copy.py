#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


forestfires = pd.read_csv("forestfires.csv")


# In[3]:


forestfires


# In[4]:


data = forestfires.describe()


# In[5]:


##Dropping the month and day columns
forestfires.drop(["month","day"],axis=1,inplace =True)


# In[6]:


forestfires


# In[7]:


predictors = forestfires.iloc[:,0:28]
target = forestfires.iloc[:,28]


# In[8]:


predictors


# In[9]:


target


# In[10]:


##Normalising the data as there is scale difference
def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return (x)


# In[11]:


fires = norm_func(predictors)


# In[12]:


fires


# In[13]:


x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size = 0.25, stratify = target)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[14]:


model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)


# In[15]:


pred_test_linear = model_linear.predict(x_test)
np.mean(pred_test_linear==y_test) # Accuracy = 98.46%


# In[16]:


acc = accuracy_score(y_test, pred_test_linear) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, pred_test_linear)


# In[17]:


# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(x_train,y_train)


# In[18]:


pred_test_poly = model_poly.predict(x_test)
np.mean(pred_test_poly==y_test)


# In[19]:


# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train,y_train)


# In[20]:


pred_test_rbf = model_rbf.predict(x_test)
np.mean(pred_test_rbf==y_test) #Accuracy = 76.15%


# In[21]:


#'sigmoid'
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(x_train,y_train)


# In[22]:


pred_test_sig = model_rbf.predict(x_test)
np.mean(pred_test_sig==y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




