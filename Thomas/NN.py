
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from collections import Counter
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.utils import shuffle


# In[3]:


test = pd.read_csv('../output_preprocessed/test_last_transaction.csv')
test_features = test[test.columns[2:]]

predictions = pd.DataFrame()
predictions['msno']=test['msno']


# In[4]:


train = pd.read_csv('../output_preprocessed/train_last_transaction.csv')


# In[5]:


train.is_churn


# In[6]:



# desired_apriori=0.10

# # Get the indices per target value
# idx_0 = train[train.is_churn == 0].index
# idx_1 = train[train.is_churn == 1].index



# # Get original number of records per target value
# nb_0 = len(train.loc[idx_0])
# nb_1 = len(train.loc[idx_1])

# # Calculate the undersampling rate and resulting number of records with target=0
# undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)
# undersampled_nb_0 = int(undersampling_rate*nb_0)

# # Randomly select records with target=0 to get at the desired a priori
# undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_nb_0)

# # Construct list with remaining indices
# idx_list = list(undersampled_idx) + list(idx_1)

# # Return undersample data frame
# train = train.loc[idx_list].reset_index(drop=True)


# In[7]:


print(len(train[train.is_churn == 1]))
print(len(train[train.is_churn == 0]))


# 1: 150801
# 0: 1813090

# In[8]:


train = train[train.columns[1:]]
train_features = train[train.columns[1:]]
train_labels = train["is_churn"]
del train


# In[9]:


train_features.head()


# In[10]:


(train_labels == 1).sum()/len(train_labels)


# In[11]:


train_features.columns


# In[12]:


def replacemean(x, t, mean):
    if(x < t):
        return mean
    else:
        return x


# In[13]:


mean_registration_init_time = train_features[train_features["registration_init_time"] > 1]["registration_init_time"].mean()
train_features["registration_init_time"] = train_features["registration_init_time"].apply(lambda x: replacemean(x, 1, mean_registration_init_time))
test_features["registration_init_time"] = test_features["registration_init_time"].apply(lambda x: replacemean(x, 1, mean_registration_init_time))


# In[14]:


mean_transaction_date = train_features[train_features["transaction_date"] > 1]["transaction_date"].mean()
train_features["transaction_date"] = train_features["transaction_date"].apply(lambda x: replacemean(x, 1, mean_transaction_date))
test_features["transaction_date"] = test_features["transaction_date"].apply(lambda x: replacemean(x, 1, mean_transaction_date))


# In[15]:


mean_date = train_features[train_features["date"] > 1]["date"].mean()
train_features["date"] = train_features["date"].apply(lambda x: replacemean(x, 1, mean_date))
test_features["date"] = test_features["date"].apply(lambda x: replacemean(x, 1, mean_date))


# In[16]:


cols_to_transform = ["city", "gender", "payment_method_id"]

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
enc.fit(train_features.loc[:][cols_to_transform])

ONE_HOT_train = (enc.transform(train_features.loc[:][cols_to_transform]).toarray()).transpose()
ONE_HOT_test = (enc.transform(test_features.loc[:][cols_to_transform]).toarray()).transpose()

for col in cols_to_transform:
    del train_features[col]
    del test_features[col]

for i in range(0, ONE_HOT_train.shape[0]):
    train_features["ONE_HOT_"+str(i)] = ONE_HOT_train[i]
    test_features["ONE_HOT_"+str(i)] = ONE_HOT_test[i]


# In[17]:


from sklearn.preprocessing import Normalizer
norm = Normalizer()
norm.fit(train_features)

train_features_preprocessed = norm.transform(train_features)
test_features_preprocessed = norm.transform(test_features)


# In[18]:


def compute_logloss(ischurn, pred_proba):
    logloss = -((ischurn*np.log(pred_proba)).sum() + ((1 - ischurn)*np.log(1 - pred_proba)).sum())
    return (logloss / len(pred_proba))


# In[19]:


class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]

                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train)
#                cross_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
#                print("    cross_score: %.5f" % (cross_score.mean()))
                y_pred = clf.predict_proba(X_holdout)[:,1]                
                print(compute_logloss(y_pred,y_holdout))
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
            S_test[:, i] = S_test_i.mean(axis=1)

#         results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')
#         print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)
        return res


# In[20]:


from xgboost import XGBClassifier
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


from keras.models import Sequential
from keras.layers import Dense
# fix random seed for reproducibility
numpy.random.seed(7)
1
2
3
4
5
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
np.random.seed(7)

import lightgbm as lgb


K = 5
# kf = KFold(n_splits=K)
kf = StratifiedKFold(n_splits=K)

#gbm = XGBClassifier(max_delta_step = 1)
#gbm.fit(train_features_preprocessed, train_labels)
#predictions["pred_gbm"] = gbm.predict_proba(test_features_preprocessed)[:,1]


# lgbm = lgb.LGBMClassifier(n_estimatorsobjective='binary')
# lgbm.fit(train_features_preprocessed, train_labels)
# predictions["pred_lgbm"] = lgbm.predict_proba(test_features_preprocessed)[:,1]



#adb = ensemble.AdaBoostClassifier()
#adb.fit(train_features_preprocessed, train_labels)
#predictions["pred_adb"] = adb.predict_proba(test_features_preprocessed)[:,1]


#lg = LogisticRegression()
#lg.fit(train_features_preprocessed, train_labels)
#predictions["pred_lg"] = lg.predict_proba(test_features_preprocessed)[:,1]



# In[21]:


train_features_preprocessed.shape[1]


# In[83]:


#max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_alpha =2, reg_lambda = 6,

layer1=24
layer2=24
layer3=1

model = Sequential()
model.add(Dense(24, input_dim=train_features_preprocessed.shape[1], activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[compute_logloss])
targets = np.asarray([0.]*test.shape[0])
for train_index,test_index in kf.split(train_features_preprocessed,train_labels):
    X_train, y_train = train_features_preprocessed[train_index].copy(),train_labels[train_index].copy()
    X_test, y_test = train_features_preprocessed[test_index].copy(),train_labels[test_index].copy()
    fit_model = model.fit(X_train, y_train, epochs=150, batch_size=10)
    y_predict=fit_model.predict_proba(X_test)
    targets += fit_model.predict_proba(test_features_preprocessed)[:,1]
    del y_test, X_train, X_test, y_train
targets /=K
test_target = pd.DataFrame()
test_target['msno']=test['msno']
test_target['is_churn']=targets
test_target.to_csv('../output/output_NN_l1/'+str(layer1)+'_l2/'+str(layer2)+'_l3/'+str(layer3)+'.csv', index=False)
#predictions.to_csv("../output/from_kaggle_kernels_and_calculs.csv", float_format='%.6f', index = False)


# In[ ]:


# cl1 = lgb.LGBMClassifier(n_estimators=1000, learning_rate =0.005 , max_depth=-1,subsample = 1, colsample_bytree=0.9,objective='binary')
# cl2 = lgb.LGBMClassifier(n_estimators=200, learning_rate =0.09 , max_depth=-1, subsample = 0.9, colsample_bytree=0.9,objective='binary')

# lr1 = LogisticRegression()
# stack = Ensemble(n_splits=5,
#         stacker = lr1,
#         base_models = (cl1,cl2))

# targets = stack.fit_predict(train_features_preprocessed,train_labels,test_features_preprocessed)[:,1]
# test_target = pd.DataFrame()
# test_target['msno']=test['msno']
# test_target['is_churn']=targets


# In[90]:


# test_target.to_csv('../output/output_LGB_cl1_cl2.csv', index=False)


# /// est=200, et=0.09, depth=-1, col=0.9, sub=0.9/////
# 0.0789238541209
# 0.0796247254159
# 0.120800671696
# 0.125796860132
# 0.126266228716
# /// est=1000, et=0.005, depth=-1, col=1, sub=0.8/////
# 0.0831707967063
# 0.0840120852989
# 0.130570757225
# 0.136304585383
# 0.136338357307
# /// est=1000, et=0.005, depth=-1, col=0.9, sub=1/////
# 0.0833001724272
# 0.084050353593
# 0.130303875663
# 0.136283216618
# 0.136293672491
# ///////////////////////////////////////////////////////////////////////////////
# est=1000 et=0.005 depth=-1
# 0.0832268324013
# 0.0840225550256
# 0.130625156578
# 0.136549348377
# 0.136210738066
# ///////////////////////////////////////////////////////////////////////////////
# est=1000 et=0.005 depth=7
# 0.0831035400493
# 0.0839660360624
# 0.132396293088
# 0.138398769378
# 0.138883725537
# 
# 
