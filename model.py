import pandas as pd
from tqdm import tqdm

import numpy as np
from sklearn.preprocessing import StandardScaler
import time
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC,LinearSVR
import lightgbm as lgb

from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import gc

start_time = time.time()
PROCESSED_DATA = 'data/processed_data'
SUBMISSION_FILE = 'submissions/stacking.csv'
train = pd.read_csv(PROCESSED_DATA +'train_features.csv')
test = pd.read_csv(PROCESSED_DATA +'test_features.csv')


VALIDATION = False
dates_train = train['date_block_num']

dates_train_level2 = dates_train[dates_train.isin([27, 28, 29, 30, 31, 32])].unique()


def downcast_types(df):


    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype == "int64"]
    # Downcast
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int32)
    return df





to_drop = ['target','date_block_num']
mean_enc_cols = [col for col in train.columns if 'mean_enc'  in col]
for col in mean_enc_cols:
    test.fillna(train[col].mean(),inplace=True)

lgb_params = {
                   'feature_fraction': 0.75,
                   'metric': 'rmse',
                   'nthread':1,
                   'min_data_in_leaf': 2**7,
                   'bagging_fraction': 0.75,
                   'learning_rate': 0.03,
                   'objective': 'mse',
                   'bagging_seed': 2**7,
                   'num_leaves': 2**7,
                   'bagging_freq':1,
                   'verbose':0
                  }

svm = LinearSVR(C=1.0,verbose=True)
scaler = StandardScaler()

train['target'] = train['target'].clip(0,20)



print('--------------- Scaling Features --------------')
x_train = train.drop(to_drop,axis=1)
x_train = downcast_types(x_train)
x_train = scaler.fit_transform(x_train)
y_train = train['target']
test.drop(to_drop,axis=1,inplace=True)
test = downcast_types(test)
test = scaler.transform(test)

del train
gc.collect()

print('---------- First level models ---------------')

lr = LinearRegression()
lr.fit(x_train,y_train)
pred_lr = lr.predict(test)

lgb_params = {
               'feature_fraction': 0.75,
               'metric': 'rmse',
               'nthread':1,
               'min_data_in_leaf': 2**7,
               'bagging_fraction': 0.75,
               'learning_rate': 0.03,
               'objective': 'mse',
               'bagging_seed': 2**7,
               'num_leaves': 2**7,
               'bagging_freq':1,
               'verbose':0
              }

def baseline_model(n_features):
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=n_features, kernel_initializer='uniform', activation='softplus'))
    model.add(Dense(1, kernel_initializer='uniform', activation='relu'))
    # Compile model
    model.compile(loss='mse', optimizer='Nadam', metrics=['mse'])
    # model.compile(loss='mean_squared_error', optimizer='adam')
    return model


kr_reg = KerasRegressor(build_fn=baseline_model,
                        n_features=x_train.shape[1],
                        verbose=1,epochs = 5,batch_size = 5500)
lgb_est = lgb.LGBMRegressor(**lgb_params)
knn = KNeighborsRegressor(n_jobs=5,n_neighbors=10)

estimators = [lr,lgb_est,knn,kr_reg]
X_test_level2 = np.zeros(shape=(len(test),len(estimators)))

for est_index,estimator in enumerate(estimators):
    estimator.fit(x_train,y_train)
    X_test_level2[:,est_index] = estimator.predict(test)

lgb_est.fit(x_train,y_train)
pred_lgb = lgb_est.predict(test)









sgd = SGDRegressor(tol=1e-3,max_iter=1000)

print('---------------- Fitting a model ---------')

X_train_level2 = np.zeros(shape=(x_train[dates_train.isin(dates_train_level2)].shape[0],len(estimators)))
y_train_level2 = y_train[dates_train.isin(dates_train_level2)]

index = 0

# And here we create 2nd level feeature matrix, init it with zeros first
X_train_level2 = np.zeros([y_train_level2.shape[0], 2])

# Now fill `X_train_level2` with metafeatures
index = 0
for cur_block_num in tqdm([27, 28, 29, 30, 31, 32]):
    X_train_1, y_train_1 = x_train[dates_train < cur_block_num], y_train[dates_train < cur_block_num]

    X_val_2, y_val_2 = x_train[dates_train == cur_block_num], y_train[dates_train == cur_block_num]
    # idx = np.where(dates_train == cur_block_num)
    idx = X_val_2.shape[0]
    for est_idx,estimator in enumerate(estimators):
        estimator.fit(X_train_1,y_train_1)
        X_train_level2[index:index + idx,est_idx] = estimator.predict(X_val_2)
        estimator.predict(X_val_2)
    '''
    lr_pred = lr.predict(X_val_2)
    model = lgb.train(lgb_params, lgb.Dataset(X_train_1, label=y_train_1), 100)
    pred_lgb = model.predict(X_val_2)
    X_train_level2[index:index + idx, 0] = lr_pred
    X_train_level2[index:index + idx, 1] = pred_lgb
    '''
    index += idx


sgd.fit(X_train_level2,y_train_level2)
pred = sgd.predict(X_test_level2)
pred = np.round(pred,4)


submission = pd.DataFrame({'ID' : range(len(pred)),'item_cnt_month': pred})
submission.to_csv(SUBMISSION_FILE,index=False)
print('Process Complete {:.4f}'.format((time.time() - start_time)/60))


print('Fitting models complete {:.2f}'.format((time.time() -start_time)/60))








