import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor


pd.set_option('display.max_columns', None)  # 设置显示所有列
data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')
data_prepare = data_train.append(data_test)
data_prepare.reset_index(drop=True, inplace=True)

data_prepare['year'] = data_prepare.datetime.apply(lambda x:x.split('-')[0])
data_prepare['month'] = data_prepare.datetime.apply(lambda x:x.split('-')[1])
data_prepare['weekday'] = data_prepare.datetime.apply(lambda x:datetime.strptime(x.split()[0],'%Y-%m-%d').weekday())
data_prepare['hour'] = data_prepare.datetime.apply(lambda x:x.split()[1].split(':')[0])

# 对缺失的风速进行填补
data_wind = data_prepare.loc[data_prepare.windspeed != 0]
data_nowind = data_prepare.loc[data_prepare.windspeed == 0]
data_wind_X = data_wind.loc[:,['season','weather','temp','atemp','humidity','year','month','hour']]
data_wind_y = data_wind['windspeed']
data_nowind_X = data_nowind.loc[:,['season','weather','temp','atemp','humidity','year','month','hour']]
wind_rfr = RandomForestRegressor()
wind_rfr.fit(data_wind_X, data_wind_y)
data_nowind['windspeed'] = wind_rfr.predict(data_nowind_X)

data_prepare = data_wind.append(data_nowind)
data_prepare.reset_index(drop=True, inplace=True)

for item in ['season','weather','year','month','weekday','hour']:
    data_prepare[item] = data_prepare[item].astype('category')

data_prepare_train = data_prepare.loc[pd.notnull(data_prepare['count'])]
data_prepare_test = data_prepare.loc[pd.isnull(data_prepare['count'])]
data_prepare_train_y = data_prepare_train.loc[:,'count']
data_prepare_train_X = data_prepare_train.drop(columns=['datetime','casual','registered','count'])
data_prepare_test_X = data_prepare_test.drop(columns=['datetime','casual','registered','count'])
# print(data_prepare_train.info())
# print(data_prepare_test.info())
# print(data_prepare.info())
print(data_prepare_train_X.head(5))
print(data_prepare_test_X.head(5))