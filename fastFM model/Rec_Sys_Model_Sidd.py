#!/usr/bin/env python
# coding: utf-8

# # Importing libraries


import numpy as np
import pandas as pd

import sklearn 

import os
# import tensorflow as tf


data_dir = './data/'

csv_files = os.listdir(data_dir)


df_di = {}

for file in csv_files:
    df_di[file] = pd.read_csv(os.path.join(data_dir,file))



df_di.keys()



main_df = df_di['df.csv']

df_di['df.csv'].head()


len(main_df)


main_df.describe()


u_watch_df = df_di['User_Watched.csv']


u_watch_df.head()


club_df = pd.merge(main_df,u_watch_df,on=['UserID','MovieID'])

club_df.head()


len(club_df)



import scipy




main_df['Days_Since_Watched'].min(),main_df['Days_Since_Watched'].max()


# ## Building the data matrix

data = []


mi = main_df['Days_Since_Watched'].min()
ma = main_df['Days_Since_Watched'].max()

for u_id,h_id,d_sw in zip(main_df['UserID'],main_df['MovieID'],main_df['Days_Since_Watched']):
    scaled_d_sw = int(((d_sw-mi) /(ma-mi))*10)
    
    di = {'u_id':u_id,'h_id':h_id,'day_bin': scaled_d_sw}
    
    data.append(di)
    

print(len(data))


u_ids,h_ids = {},{}

for d in data:
    u,h,da = d['u_id'],d['h_id'],d['day_bin']
    
    if not u in u_ids: u_ids[u] = len(u_ids)
    if not h in h_ids: h_ids[h] = len(h_ids)
        
        
nUsers,nHotels = len(u_ids),len(h_ids)


print(nUsers,nHotels)



nDays = 10 - 0 + 1




X = scipy.sparse.lil_matrix((len(data), nUsers + nHotels + nDays))


for i in range(len(data)):
    user = u_ids[data[i]['u_id']]
    hotel = h_ids[data[i]['h_id']]
    days = data[i]['day_bin'] 
    
    X[i,user] = 1 # One-hot encoding of user
    X[i,nUsers + hotel] = 1 # One-hot encoding of hotels
    X[i,nUsers + nHotels + days] = 1

y = np.array([d for d in club_df['Number_Watched_log']])

y[:10]


X.shape[0],X.shape[0]*0.7



X_train,y_train = X[:40773],y[:40773]
X_test,y_test = X[40773:],y[40773:]

def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


from fastFM import als


fm = als.FMRegression(n_iter=1000, init_stdev=0.1, rank=5, l2_reg_w=0.1, l2_reg_V=0.5)


fm.fit(X_train, y_train)

y_pred_with_features = fm.predict(X_test)


import math

print("rmse: ",math.sqrt(MSE(y_pred_with_features, y_test)))



### 2.316082581455804

