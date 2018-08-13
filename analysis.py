import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re
from ggplot import *
import itertools
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold,train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

DATA_DIR = 'data/'
items = pd.read_csv(DATA_DIR + 'items.csv')
item_cat = pd.read_csv( DATA_DIR + 'item_categories.csv')
shops = pd.read_csv(DATA_DIR + 'shops.csv')
train = pd.read_csv(DATA_DIR + 'sales_train.csv.gz')
test = pd.read_csv(DATA_DIR + 'test.csv.gz')

'''
Creating aggregate data as indicated in mean encoding assignment 
'''

index_cols = ['shop_id', 'item_id', 'date_block_num']
for date_block in train['date_block_num'].unique():
    shop_ids = train.loc[train.date_block_num == date_block,'shop_id'].unique()
    item_ids = train.loc[train.date_block_num == date_block,'item_id'].unique()
    prds = itertools.product(*[shop_ids,item_ids,[date_block]])

gb = train.groupby(index_cols,as_index=False).agg({'item_cnt_day':{'target':'sum'}})
#pd.DataFrame(list(prds),columns=index_cols)
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
grid = pd.DataFrame(np.vstack(prds),columns=index_cols,dtype=np.int32)
all_data = pd.merge(grid,gb,how='left',on=index_cols).fillna(0)
#sort the data
all_data.sort_values(['date_block_num','shop_id','item_id'],inplace=True)

# Aggregating and having constant predictions
mean_values = all_data.groupby(['shop_id','item_id'])['target'].mean()

# Creating baseline prediction
baseline_df = pd.merge(test,mean_values.reset_index(),on=['shop_id','item_id'],how='left')

# Filling missing values with 0.3343
baseline_df['target'] = baseline_df['target'].fillna(0.3343)
baseline_df[['ID','target']].rename(columns={'target': 'item_cnt_month'}).to_csv('submissions/baseline_submission.csv',index=False)

# Mean encoding

all_data = all_data.reset_index()

# Applying kfold mean encoding
cat_cols = ['shop_id','item_id']
kfolds = KFold(n_splits=5,shuffle=False)
for trn_idx,val_idx in kfolds.split(all_data):
    for col in cat_cols:
        X_trn,X_val = all_data.iloc[trn_idx],all_data.iloc[val_idx]
        cat_encoder = all_data.groupby(col)['target'].transform(np.mean)
        all_data.loc[val_idx,col + '_mean_enc'] = X_val[col].map(cat_encoder)


test_df = pd.merge(test,all_data[['shop_id','item_id','shop_id_mean_enc','item_id_mean_enc']],on=['shop_id','item_id'],how='left')
test_df['item_id_mean_enc'].fillna(all_data['item_id_mean_enc'].mean(),inplace=True)
test_df['shop_id_mean_enc'].fillna(all_data['shop_id_mean_enc'].mean(),inplace=True)

rf = RandomForestRegressor(criterion='mse')
X_train,X_test,Y_train,Y_test = train_test_split(all_data.drop('target',axis=1),all_data['target'],test_size = -.2)
rf_mdl = rf.fit(X_train.loc[:,['shop_id_mean_enc','item_id_mean_enc']].values,Y_train.values)
pred = rf_mdl.predict(X_test.loc[:,['shop_id_mean_enc','item_id_mean_enc']].values)
rf_mdl.score(X_test.loc[:,['shop_id_mean_enc','item_id_mean_enc']].values,Y_test)

# Applying on test
test_df['item_cnt_month'] = rf_mdl.predict(test_df.loc[:,['shop_id_mean_enc','item_id_mean_enc']].values)

test_df[['ID','item_cnt_month']].to_csv('submissions/mean_encoding.csv',index=False)
