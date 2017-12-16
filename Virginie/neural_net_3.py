
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.utils import np_utils

from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
np.random.seed(123)


# In[2]:


train_set = pd.read_csv("../output_preprocessed/train_lastandmean_transaction.csv")
# split into input (X) and output (Y) variables
X = train_set.drop(["is_churn",'msno'], axis=1)
Y = train_set['is_churn']


# In[4]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    train_size=0.75,
                                                    test_size=0.25)


# In[11]:


X_train = X_train.values
Y_train = Y_train.values


# In[6]:


input_dim = X_train.shape[1]
encoding_dim = 14


# In[7]:


test_set = pd.read_csv("../output_preprocessed/test_lastandmean_transaction.csv")
results = pd.DataFrame()
results['is_churn'] = test_set['is_churn']
test_set = test_set.drop(['msno', 'is_churn'], axis=1)


# In[13]:


model = Sequential()
# Add a dropout layer for input layer
model.add(Dropout(0.2, input_shape=input_dim))

# Add fully connected layer with a ReLU activation function
model.add(Dense(units=16, activation='relu'))

# Add a dropout layer for previous hidden layer
model.add(Dropout(0.5))

# Add fully connected layer with a ReLU activation function
model.add(Dense(units=16, activation='relu'))

# Add a dropout layer for previous hidden layer
model.add(Dropout(0.5))

# Add fully connected layer with a sigmoid activation function
model.add(Dense(units=1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


# Fit the model
model.fit(X_train, Y_train, epochs=20, batch_size=1,  verbose=2)


# In[ ]:


model.save('model_2.h5')


# In[ ]:


score = model.evaluate(X_test, Y_test, verbose=0)
print(score)


# In[ ]:


# calculate predictions
predictions = model.predict_proba(test_set)[:,1]
results['is_churn']=predictions


# In[ ]:


results.to_csv("../output/from_keras.csv", float_format='%.6f', index = False)

