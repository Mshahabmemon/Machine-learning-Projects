#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[6]:


df = pd.read_csv('C:\\Users\\shaha\\Documents\\University US\\Business analytics\\udemy\\Refactored_Py_DS_ML_Bootcamp-master\\15-Decision-Trees-and-Random-Forests\\loan_data.csv')


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.head(5)


# In[10]:


plt.figure(figsize = (9,5))
df[df['credit.policy'] == 1]['fico'].hist(alpha = 0.5, color = 'red', bins = 30)
df[df['credit.policy'] == 0]['fico'].hist(alpha = 0.5, color = 'blue', bins = 30)


# In[11]:


plt.figure(figsize = (9,5))
df[df['not.fully.paid'] == 1]['fico'].hist(alpha = 0.5, color = 'red', bins = 30)
df[df['not.fully.paid'] == 0]['fico'].hist(alpha = 0.5, color = 'red', bins = 30)
plt.legend()
plt.xlabel('FICO')


# In[12]:


sns.countplot(x = 'purpose', data = df, hue = 'not.fully.paid')


# In[13]:


sns.jointplot(x = 'fico', y = 'int.rate', data = df)


# In[14]:


plt.figure(figsize = (10,6))
sns.lmplot(y='int.rate',x='fico',data=df,hue='credit.policy',
           col='not.fully.paid',palette='Set1')


# In[15]:


df.info()


# In[16]:


cat_feats = ['purpose']


# In[17]:


df = pd.get_dummies(data = df, columns = cat_feats, drop_first= True)


# In[18]:


from sklearn.model_selection import train_test_split


# In[20]:


X = df.drop('not.fully.paid', axis = 1)
y = df['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101) 


# In[26]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


# In[22]:


dForest = RandomForestClassifier()


# In[23]:


dForest.fit(X_train, y_train)


# In[24]:


predictions1 = dForest.predict(X_test)


# In[27]:


print(classification_report(y_test, predictions1))


# In[29]:


print(confusion_matrix(y_test, predictions1))


# In[ ]:




