import pandas as pd
from tqdm import tqdm

import numpy as np
from sklearn.preprocessing import StandardScaler
import time
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC,LinearSVR
from scipy.sparse import csr_matrix
import lightgbm as lgb

from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import gc

start_time = time.time()
PROCESSED_DATA = 'data/processed_data'
SUBMISSION_FILE = 'submissions/lgb_submission.csv'
train = pd.read_csv(PROCESSED_DATA +'train_features.csv')
test = pd.read_csv(PROCESSED_DATA +'test_features.csv')


VALIDATION = False
dates_train = train['date_block_num']
last_date = 33
dates_test = [30,31,32,33]
to_drop = ['target','date_block_num']

x_train = train[dates_train < last_date].drop(to_drop,axis=1)
y_train = train.loc[dates_train < last_date,'target']
x_test = train.loc[dates_train == last_date].drop(to_drop,axis=1)
y_test = train.loc[dates_train == last_date,'target']

x_train = csr_matrix.tocsr(x_train)
x_test = csr_matrix.tocsr(x_test)


lgb_params = {'boosting_type': 'gbdt',
          'max_depth' : -1,
          'objective': 'mse',
          'nthread': 3, # Updated from nthread
          'num_leaves': 128,
          'learning_rate': 0.1,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'reg_alpha': 5,
          'reg_lambda': 10,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'feature_fraction': 0.8,
          'num_class' : 1,
          'metric' : 'mse'}


grid_params = gridParams = {
    'learning_rate': [0.1],
    'n_estimators': [40,100,150],
    'num_leaves': [128],
    'boosting_type' : ['gbdt'],
    'objective' : ['mse'],
    'random_state' : [501], # Updated from 'seed'
    'colsample_bytree' : [0.65, 0.66],
    'subsample' : [0.7,0.75],
    'reg_alpha' : [1,1.2],
    'reg_lambda' : [1,1.2,1.4],
    }


feature_factions = [0.65,0.7,0.75,0.80,0.9]
num_leaves = [64,128,256]
lgb = lgb.LGBMRegressor(**lgb_params)

'''
grid = GridSearchCV(lgb,grid_params,verbose=0,cv=4,n_jobs=4)

# Run the grid
grid.fit(x_train, y_train)
# Printing best parameters and scores
print(grid.best_params_)
print(grid.best_score_)

lgb_params['reg_alpha'] = grid.best_params['reg_alpha']
lgb_params['reg_lambda'] = grid.best_params['reg_lambda']
lgb_params['colsample_bytree'] = grid.best_params['colsample_bytree']
lgb_params['colsample_bytree'] = grid.best_params['colsample_bytree']
lgb_params['n_estimators'] = grid.best_params_['n_estimators']
lgb.set_params(**lgb_params)
'''

X = train.drop(['target'],axis=1)
test = test.drop(['target'],axis=1)
Y = train['target'].values



lgb.fit(X,Y,verbose=False)
pred = lgb.predict(test)
print(len(pred))
submission = pd.DataFrame({'ID' : range(0,len(pred)),'item_cnt_month': pred})
submission.to_csv(SUBMISSION_FILE,index=False)
print('Process Complete {:.4f}'.format((time.time() - start_time)/60))







