

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import sklearn as sk
import scipy as sc
import os
from os.path import join
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import dask.dataframe as dd


#%%

'''
Load the data
'''

path = '/media/max/HDD3/DNS_Data/Planar/NX512/UPRIME5/postProcess_UVW'

case = '16'

data_tensor = pd.read_pickle(join(path,'filter_width_TOPHAT_%s_tensor.pkl' % case))
data_grads = pd.read_pickle(join(path,'filter_width_TOPHAT_%s_grad_LES_tensor.pkl' % case))

#%%

print(data_tensor.columns)
print(data_grads.columns)

#%%

# remove c=0 and c=1

data_tensor_dd = dd.from_pandas(data_tensor,npartitions=1)
data_grads_dd = dd.from_pandas(data_grads,npartitions=1)

data_all = dd.concat([data_grads_dd,data_tensor_dd],axis=1)

data_all = data_all[data_all['c_bar']<0.99]
data_all = data_all[data_all['c_bar']>0.4]

#data_all = data_all.compute()

#%%
# shuffle

data_all = data_all.sample(frac=1.0).reset_index(drop=True).compute()

#%%
split_index = int(len(data_all)*0.7)

Train = data_all.iloc[0:split_index]
Test = data_all.iloc[split_index:]

X_train = Train.drop(['omega_DNS_filtered','Xi_iso_085'],axis=1)
y_train = Train['omega_DNS_filtered']

X_test = Test.drop(['omega_DNS_filtered','Xi_iso_085'],axis=1)
y_test = Test['omega_DNS_filtered']

#%%
Dtrain = xgb.DMatrix(data=X_train,label=y_train)

Dtest = xgb.DMatrix(data=X_test)

#%%

param = {'max_depth': 10,
         'eta': 0.1,
         'objective': 'reg:squarederror',
         'tree_method':'exact',
         'min_child_weight':1,
         'gamma': 0.3,
          'subsample':0.6,
         'grow_policy':'depthwise'}

num_round = 10

model = xgb.train(param, Dtrain, num_round)

#%%
y_pred = model.predict(Dtest)

print(sc.stats.pearsonr(y_test,y_pred))

plt.scatter(y_pred,y_test,s=0.2)
plt.show()

xgb.plot_importance(model)
plt.show()