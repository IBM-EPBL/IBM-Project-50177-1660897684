#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical


# In[6]:


#download the data set
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[9]:


#first image in the data set
plt.imshow(X_train[5])


# In[11]:


#image shape
X_train[0].shape


# In[12]:


#reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)


# In[13]:


X_train[0].shape


# In[14]:


y_train[5]


# In[15]:


#one hot encode target  coiumn
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train[0]


# In[16]:


#create model
model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))


# In[17]:


#compile model using accuracy as a measure of model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[18]:


#train model
model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=3)


# In[19]:


#show actual results for the first 5 images in the test set
y_test[:6]


# In[ ]:




