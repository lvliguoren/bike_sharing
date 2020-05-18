import pandas as pd


pd.set_option('display.max_columns', None)  # 设置显示所有列
data_train = pd.read_csv("data/train.csv")
data_test = pd.read_csv("data/test.csv")
data_prepare = data_train.append(data_test)
print(data_prepare.head(5))
print(data_prepare.info())