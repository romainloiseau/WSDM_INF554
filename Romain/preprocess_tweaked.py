
# coding: utf-8

# In[14]:


from multiprocessing import Pool, cpu_count
import gc; gc.enable()
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import *
import sklearn

train = pd.read_csv('../input/train.csv')
train = pd.concat((train, pd.read_csv('../input/train_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)
test = pd.read_csv('../input/sample_submission_v2.csv')

transactions = pd.read_csv('../input/transactions.csv', usecols=['msno'])
transactions = pd.concat((transactions, pd.read_csv('../input/transactions_v2.csv', usecols=['msno'])), axis=0, ignore_index=True).reset_index(drop=True)
transactions = pd.DataFrame(transactions['msno'].value_counts().reset_index())
transactions.columns = ['msno','trans_count']
train = pd.merge(train, transactions, how='left', on='msno')
test = pd.merge(test, transactions, how='left', on='msno')
transactions = []; print('transaction merge...')

user_logs = pd.read_csv('../input/user_logs_v2.csv', usecols=['msno'])
#user_logs = pd.read_csv('../input/user_logs.csv', usecols=['msno'])
#user_logs = pd.concat((user_logs, pd.read_csv('../input/user_logs_v2.csv', usecols=['msno'])), axis=0, ignore_index=True).reset_index(drop=True)
user_logs = pd.DataFrame(user_logs['msno'].value_counts().reset_index())
user_logs.columns = ['msno','logs_count']
train = pd.merge(train, user_logs, how='left', on='msno')
test = pd.merge(test, user_logs, how='left', on='msno')
user_logs = []; print('user logs merge...')

members = pd.read_csv('../input/members_v3.csv')
train = pd.merge(train, members, how='left', on='msno')
test = pd.merge(test, members, how='left', on='msno')
members = []; print('members merge...') 


# In[15]:


gender = {'male':1, 'female':2}
train['gender'] = train['gender'].map(gender)
test['gender'] = test['gender'].map(gender)

train = train.fillna(0)
test = test.fillna(0)


# In[16]:


transactions = pd.read_csv('../input/transactions_v2.csv') #pd.read_csv('../input/transactions.csv')
#transactions = pd.concat((transactions, pd.read_csv('../input/transactions_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)
transactions = transactions.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)
last_transactions = transactions.drop_duplicates(subset=['msno'], keep='first')

train = pd.merge(train, last_transactions, how='left', on='msno')
test = pd.merge(test, last_transactions, how='left', on='msno')
del last_transactions

mean_transactions = transactions.groupby("msno").mean()
train = pd.merge(train, mean_transactions, how='left', left_on='msno', right_index = True)
test = pd.merge(test, mean_transactions, how='left', left_on='msno', right_index = True)
del mean_transactions
del transactions


# In[ ]:


def transform_df(df):
    df = pd.DataFrame(df)
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df

def transform_df2(df):
    df = df.sort_values(by=['date'], ascending=[False])
    #df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df

df_iter = pd.read_csv('../input/user_logs.csv', low_memory=False, iterator=True, chunksize=10000000)
last_user_logs = []
i = 0 #~400 Million Records - starting at the end but remove locally if needed
for df in df_iter:
    if i>35:
        if len(df)>0:
            print(df.shape)
            p = Pool(cpu_count())
            df = p.map(transform_df, np.array_split(df, cpu_count()))   
            df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
            df = transform_df2(df)
            p.close(); p.join()
            last_user_logs.append(df)
            print('...', df.shape)
            df = []
    i+=1

print("Fin iterations")
#last_user_logs.append(transform_df(pd.read_csv('../input/user_logs_v2.csv')))
last_user_logs.append(pd.read_csv('../input/user_logs_v2.csv'))
last_user_logs = pd.concat(last_user_logs, axis=0, ignore_index=True).reset_index(drop=True)

very_last_user_logs = transform_df2(last_user_logs)
print("Fin calcul very_last_user_logs")
train = pd.merge(train, very_last_user_logs, how='left', on='msno')
test = pd.merge(test, very_last_user_logs, how='left', on='msno')
print("Fin merge very_last_user_logs")

# last_user_logs["firstindex"] = last_user_logs["msno"].apply(lambda x: very_last_user_logs[very_last_user_logs["msno"] == x].index[0])
# last_user_logs = last_user_logs[last_user_logs["firstindex"] != last_user_logs.index]
# del last_user_logs["firstindex"]
# del very_last_user_logs
mean_user_logs = last_user_logs.groupby("msno").mean()
print("Fin calcul mean_user_logs")
train = pd.merge(train, mean_user_logs, how='left', left_on='msno', right_index = True)
test = pd.merge(test, mean_user_logs, how='left', left_on='msno', right_index = True)
print("Fin merge mean_user_logs")


# In[ ]:


train = train.fillna(0)
test = test.fillna(0)

cols = [c for c in train.columns if c not in ['is_churn','msno']]


# In[ ]:


train.to_csv("../output_preprocessed/train_lastandmean_transaction.csv", float_format='%.6f', index = False)
test.to_csv("../output_preprocessed/test_lastandmean_transaction.csv", float_format='%.6f', index = False)
print("END")


# In[6]:


import pandas as pd
df = pd.DataFrame()
df["num"] = [5,78,8,8,87,2,567,48, 4816, 45, 6, 18, 216, 8]
df["msno"] = [0, 1, 2, 3, 1, 2, 3, 4, 1, 2, 0, 1, 3, 4]
df


# In[7]:


def transform_df2(df):
    df = df.sort_values(by=['num'], ascending=[False])
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df

subdf = transform_df2(df)
subdf


# In[8]:


df["firstindex"] = df["msno"].apply(lambda x: subdf[subdf["msno"] == x].index[0])
minusdf = df[df["firstindex"] != df.index]
minusdf


# In[9]:


del df["firstindex"]
df  = df.groupby("msno").mean()
df


# In[12]:


findf = pd.merge(subdf, df, how='left', left_on='msno', right_index = True)
findf

