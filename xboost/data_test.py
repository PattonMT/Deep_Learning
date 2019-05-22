import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

csv_data = pd.read_csv('Train.csv')  # 读取训练数据
csv_test = pd.read_csv('X_test.csv') #读取测试数据
X_test = csv_test.drop('stockcode',axis=1)

X_test = X_test.drop('A_to_L_ratio',axis=1)
X_test = X_test.drop('Curr_ratio',axis=1)
X_test = X_test.drop('Cash_to_Reve',axis=1)
X_test = X_test.drop('Monetary_to_Cur',axis=1)
X_test = X_test.drop('Nonb_to_np',axis=1)
X_test = X_test.drop('ROA',axis=1)
X_test = X_test.drop('Sale_pro_ratio',axis=1)

X_train = csv_data.drop('stockcode',axis=1)# 设置 feature
X_train = X_train.drop('fake',axis=1)

X_train = X_train.drop('A_to_L_ratio',axis=1)
X_train = X_train.drop('Curr_ratio',axis=1)
X_train = X_train.drop('Cash_to_Reve',axis=1)
X_train = X_train.drop('Monetary_to_Cur',axis=1)
X_train = X_train.drop('Nonb_to_np',axis=1)
X_train = X_train.drop('ROA',axis=1)
X_train = X_train.drop('Sale_pro_ratio',axis=1)

Y_train = csv_data[['fake']] #设置 label

params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 2,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 0,
    'eta': 0.007,
    'seed': 1337,
    'nthread': 6,
}

plst = params.items()


dtrain = xgb.DMatrix(X_train, Y_train)
num_rounds = 100000
model = xgb.train(plst, dtrain, num_rounds)

# 对测试集进行预测
dtest = xgb.DMatrix(X_test)
ans = model.predict(dtest)
ans1=ans.astype(np.int)
print(ans1.dtype)
# 导出文件
code_mark=csv_test.loc[:, 'stockcode']
result_out={'stockcode':code_mark,'fake':ans1}
frame_result = pd.DataFrame(result_out)
frame_result.to_csv("./output.csv",index=False)

# 显示重要特征
plot_importance(model)
plt.show()