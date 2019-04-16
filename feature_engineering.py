import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.preprocessing import LabelEncoder
import pickle as pk
from tqdm import tqdm
import numpy as np
from itertools import product
import re
import itertools
import gc
import warnings
warnings.simplefilter("ignore",category=FutureWarning)
from sklearn.model_selection import KFold,train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
import time

validation = True
SEED = 1234

start_time = time.time()

DATA_DIR = 'data'
PROCESSED_DATA = 'data/processed_data'
items = pd.read_csv('%s/items.csv' % DATA_DIR)
item_cat = pd.read_csv( '%s/item_categories.csv' % DATA_DIR)
shops = pd.read_csv('%s/shops.csv' % DATA_DIR)
train = pd.read_csv('%s/sales_train.csv.gz' % DATA_DIR)
test = pd.read_csv('%s/test.csv.gz' % DATA_DIR)


test_nrow = test.shape[0]

print('Loading complete at {:.4f}'.format((time.time() - start_time)/60))


index_cols = ['shop_id', 'item_id', 'date_block_num']
train['item_cnt_day'] = train['item_cnt_day'].clip(0,20)
item_cnt = train.groupby(index_cols)['item_cnt_day'].agg(np.sum).reset_index().rename(columns={'item_cnt_day':'target'})





grid = []
for block_num in tqdm(train['date_block_num'].unique()):
    cur_shops = train[train['date_block_num']==block_num]['shop_id'].unique()
    cur_items = train[train['date_block_num']==block_num]['item_id'].unique()
    x = np.array(list(product(*[cur_shops, cur_items, [block_num]])))
    print(block_num,x.shape)
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

#turn the grid into pandas dataframe
index_cols = ['shop_id', 'item_id', 'date_block_num']
all_data = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)
print(all_data.shape)


all_data = pd.merge(all_data,item_cnt,on=index_cols,how='left').fillna(0)

del grid
gc.collect()
print('%0.2f min: Finish creating the grid'%((time.time() - start_time)/60))








# Aggregating shop and item level stats
shop_sales = all_data.groupby(['shop_id','date_block_num'],as_index=False)['target'].sum().\
    rename(columns={'target':'shop_month_sales'})
all_data = pd.merge(all_data,shop_sales,on = ['shop_id','date_block_num'],how='left')

item_sales = all_data.groupby(['item_id','date_block_num'],as_index=False)['target'].sum(). \
    rename(columns={'target':'item_month_sales'})
all_data = pd.merge(all_data,item_sales,on = ['item_id','date_block_num'],how='left')
del item_sales,shop_sales
gc.collect()



# Shifting feature
print('------------------ Shift features --------------')
shift_periods = [1,2,3,6,12]
test['date_block_num'] = 34
test['target'] = np.NaN
del test['ID']

all_data = pd.concat([all_data,test],axis=0)




for shift_parm in shift_periods:

    train_shift = all_data.copy()
    train_shift['date_block_num']  = train_shift['date_block_num'] + shift_parm
    train_shift = train_shift.rename(columns={'target': 'target_lag_' + str(shift_parm)})
    all_data = pd.merge(all_data,train_shift[index_cols + ['target_lag_' + str(shift_parm)]],
                        on=index_cols,how='left').fillna(0)





print('------------------ Complete shift features--------------')

train = all_data[:-test_nrow]
test = all_data[-test_nrow:]



del train_shift
del all_data
gc.collect()

train = pd.merge(train,items[['item_id','item_category_id']],on='item_id',how='left')
test = pd.merge(test,items[['item_id','item_category_id']],on='item_id',how='left')




cat_cols = ['shop_id','item_id','item_category_id']
mean_enc_cols = [col + '_mean_enc' for col in cat_cols]
# Mean encoding of categorical columns
print('------------------ Mean Encoding --------------')


kf = KFold(n_splits=5)
for trn_idx, val_idx in kf.split(train):
    for col in cat_cols:
        X_trn, X_val = train.iloc[trn_idx], train.iloc[val_idx]
        cat_encoder = train.groupby(col)['target'].transform(np.mean)
        train.loc[val_idx, col + '_mean_enc'] = X_val[col].map(cat_encoder)



mean_grp = train.groupby(['shop_id', 'item_id','item_category_id'])['shop_id_mean_enc',
                                'item_id_mean_enc','item_category_id_mean_enc'].mean()
mean_grp.reset_index(inplace=True)

mean_grp.columns = ['shop_id','item_id','item_category_id','shop_id_mean_enc',
                                'item_id_mean_enc','item_category_id_mean_enc']


test = pd.merge(test,mean_grp,on=['shop_id', 'item_id','item_category_id'], how='left')

test[mean_enc_cols].fillna(train[mean_enc_cols].mean(),inplace = True)


print('Process complete in {0:.2f}'.format((time.time() - start_time)/60))



train.to_csv('%s/train_features.csv' %PROCESSED_DATA,index=False)
test.to_csv('%s/test_features.csv' %PROCESSED_DATA,index=False)
#pk.dump(train,file=PROCESSED_DATA + open('train_features','wb'))
#pk.dump(test,file=PROCESSED_DATA + open('test_features','wb'))





















