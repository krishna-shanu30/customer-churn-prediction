#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
df= pd.read_csv(r"D:\Machine learning\datasets\Churn_Modelling.csv")


# In[44]:


df.head()


# In[45]:


df.info()


# In[46]:


# tell us about duplicated rows
df.duplicated().sum()


# In[47]:


df['Exited'].value_counts()


# In[48]:


# lets see classification in geography
df['Geography'].value_counts()


# In[49]:


# we are removing first 3 cols
df.drop(columns=['RowNumber','CustomerId','Surname'],inplace= True)
# we have taken inplace = True, which means that changes are made truely
df.head()


# In[50]:


# ohe
df= pd.get_dummies(df, columns= ['Geography','Gender'],drop_first= True,dtype=int)
# we have taken dtype= int because we donot want it on boolean values


# In[51]:


df


# In[55]:


x= df.drop(columns=['Exited'])
y= df.iloc[:,-1]


# In[56]:


from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test= tts(x,y,test_size=0.2,random_state=1)


# In[58]:


x_train.shape


# In[59]:


from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
x_train_scale= scale.fit_transform(x_train)
x_test_scale= scale.transform(x_test)


# In[60]:


import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# In[61]:


model = Sequential()
#hidden
model.add(Dense(3, activation= 'sigmoid',input_dim=11))
# output layer
model.add(Dense(1,activation='sigmoid',))


# In[62]:


model.summary()


# In[63]:


model.compile(loss='binary_crossentropy',optimizer = 'Adam')


# In[64]:


model.fit(x_train_scale,y_train,epochs=10)
# we can adjust epochs based on loss


# In[65]:


model.layers[0].get_weights()


# ## weights
# [array([[ 2.7333502e-02,  1.6617956e-02, -7.2946777e-03],
#         [ 1.7464938e-02, -3.7580680e-02, -1.5709573e-02],
#         [ 2.6588380e-02,  5.0540224e-02,  1.9132622e-03],
#         [-2.1303209e-02, -3.9278291e-02,  3.2871366e-02],
#         [ 3.4295622e-02, -4.3234769e-02,  1.6644644e-02],
#         [ 2.9223286e-02,  1.8230338e-02,  1.9314608e-02],
#         [-3.4963459e-02,  1.8682433e-02,  5.2678483e-03],
#         [ 7.0656545e-04, -3.7558372e-03, -1.6258132e-03],
#         [ 4.0838499e-02,  3.5398468e-02, -1.5331830e-02],
#         [-9.4707906e-03,  1.9718649e-02,  1.2537301e-02],
#         [-1.5348760e+00,  1.6084471e+00,  2.0973666e+00]], dtype=float32),
#  ## bias
#  array([ 0.13392426, -0.10146529, -0.01779306], dtype=float32)]
#  
# ## you can adjust layers based on which layers you want to access

# In[69]:


y_log=model.predict(x_test_scale)


# In[70]:


y_log
# now have to convert into probaility
# we will decide some treshold


# In[72]:


import numpy as np
# we have decided 0.5 treshold
# if >0.5 then 1
# else 0
y_pred= np.where(y_log>0.5,1,0)


# In[73]:


from sklearn.metrics import accuracy_score
print("Accuracy Score:", accuracy_score(y_test,y_pred))


# In[82]:


import joblib
model.save('churn_model.keras') # scaler native format
joblib.dump(scale,'scaler.joblib')
joblib.dump(list(x.columns),'features.joblib')
print("Saved: churn_model.keras, scaler.joblib, features.joblib")

