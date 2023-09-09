#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from sklearn.datasets import fetch_openml


# In[3]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# In[4]:


mnist=fetch_openml("mnist_784")


# In[5]:


mnist


# In[6]:


x,y=mnist["data"],mnist["target"]


# In[7]:


x.shape


# In[8]:


y.shape


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


import matplotlib 


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


some_digit = x.to_numpy()[36001]
some_digit_image = some_digit.reshape(28, 28)  # let's reshape to plot it

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
           interpolation='nearest')
plt.show()


# In[13]:


some_digit = x.to_numpy()[36001]
some_digit_image = some_digit.reshape(28, 28)  # let's reshape to plot it

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,interpolation='nearest')
plt.axis("off")
plt.show()


# In[14]:


y[36001]  #label of this image 


# In[15]:


x_train, x_test = x[:60000], x[6000:70000]
y_train, y_test = y[:60000], y[6000:70000]

shuffle_index = np.random.permutation(60000)


# In[16]:


x_train = x.iloc[shuffle_index]
x_test = x.iloc[shuffle_index]
y_train = y.iloc[shuffle_index]
y_test = y.iloc[shuffle_index]


# ## Creating a two detector

# In[17]:


# y_train = y_train.astype(np.int8)
# y_test = y_test.astype(np.int8)
y_train_2 = (y_train == '2')
y_test_2 = (y_test == '2')


# In[18]:


y_train


# In[19]:


y_train_2


# In[20]:


from sklearn.linear_model import LogisticRegression


# In[21]:


clf=LogisticRegression(tol=0.1) # initialize the model


# In[22]:


clf.fit(x_train,y_train_2)


# In[23]:


y.pred=clf.predict([some_digit])


# ## cross validation 

# In[24]:


from sklearn.model_selection import cross_val_score


# In[25]:


cross_val_score(clf,x_train,y_train_2,cv=3,scoring="accuracy")


# In[26]:


a=cross_val_score(clf,x_train,y_train_2,cv=3,scoring="accuracy")


# In[27]:


a.mean()

