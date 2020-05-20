import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def rmsle(y, y_,convertExp=True):
    if convertExp:
        y = np.exp(y),
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))


def plot_learing_curves(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_errors, val_errors = [],[]
    for i in range(1,len(X_train)):
        model.fit(X_train[:i],y_train[:i])
        predict_train = model.predict(X_train)
        predict_test = model.predict(X_test)
        train_errors.append(rmsle(predict_train, y_train))
        val_errors.append(rmsle(predict_test,y_test))
    plt.plot(range(1,len(X_train)),train_errors,"b--",label="Train RMSLE")
    plt.plot(range(1,len(X_train)),val_errors,"g-", label="Test RMSLE")
    plt.xlabel("Sample Nums")
    plt.ylabel("RMSLE")
    plt.legend(loc="best")
    plt.show()


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

data_rfr = RandomForestRegressor()
plot_learing_curves(data_rfr, data_prepare_train_X, data_prepare_train_y)

# print(data_prepare_train.info())
# print(data_prepare_test.info())
# print(data_prepare.info())
# print(data_prepare_train_X.head(5))
# print(data_prepare_test_X.head(5))