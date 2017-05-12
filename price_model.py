#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : price_model.py
# @Time    : 2017-05-12 11:07
# @Author  : zhang bo
# @Note    : 采用模型进行SalePrice的预测

"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import os

os.chdir('D:\PythonProjects\HousePrices\data')  # 锁定当前文件目录
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
all_data = pd.concat((train.loc[:, 'MSSubClass': 'SaleCondition'], test.loc[:, 'MSSubClass': 'SaleCondition']))


# 将numeric特征进行log化
def log_numeric(all_data):
    sale_price = pd.DataFrame({'price': train['SalePrice'], 'log(price)': np.log(train['SalePrice']),
                               'log(price+1)': np.log(train['SalePrice'] + 1), 'log1p(price)':np.log1p(train['SalePrice'])})
    sale_price.hist()
    # plt.show()
    # 1. 先对SalePrice进行log化
    train['SalePrice'] = np.log(train['SalePrice'])
    # 2. 对非object特征进行log化
    numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index  # 只有数字特征才需要log化
    skew_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))  # 先计算每个数字特征的偏度，不符合正态分布的才需log化
    skew_feats = skew_feats[skew_feats > 0.75].index  # 对偏度 > 0.75的进行log化
    all_data[skew_feats] = np.log(all_data[skew_feats])  # 对符合条件的特征进行log化
    all_data = pd.get_dummies(all_data)
    all_data = all_data.fillna(all_data.mean())  # fill na values with mean
    X_train = all_data[:train.shape[0]]  # 训练集
    X_test = all_data[train.shape[0]:]  # 测试集
    Y_train = train['SalePrice']  # target集
    return X_train, Y_train, X_test

'''model选择'''
X_train, Y_train, X_test = log_numeric(all_data)


# 计算均方误差
def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, Y_train, scoring='neg_mean_squared_error', cv=5))
    return rmse


# 岭回归
def ridge():
    model_ridge = Ridge()  # 岭回归
    alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
    cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean() for alpha in alphas]
    cv_ridge = pd.Series(cv_ridge, index=alphas)
    cv_ridge.plot(title='validation')  # rmse随alpha的变化趋势,最小的rmses是0.12733734668670765，alpha=10
    plt.xlabel('alpha')
    plt.ylabel('rmse')
    plt.show()


# L1正则化
def lasso():
    alphas = [1, 0.1, 0.001, 0.0005]
    model_lasso = LassoCV(alphas=alphas).fit(X_train, Y_train)  # 加L1正则化LASSO
    cv_lasso = rmse_cv(model_lasso)  # rsme=0.11206520404299321
    cv_lasso = pd.Series(cv_lasso, index=[1, 0.1, 0.001, 0.0005, 0.0001])
    cv_lasso.plot()  # 直线
    plt.show()
    return model_lasso

# 使用xgboost
def xgboost():
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test)
    params = {'max_depth': 2, 'eta': 0.1}
    model = xgb.cv(params, dtrain, num_boost_round=500, early_stopping_rounds=100)
    # model = xgb.train(params, dtrain, num_boost_round=500, early_stopping_rounds=100)
    model.loc[30:, ["test-rmse-mean", "train-rmse-mean"]].plot()  # 此时效果最好
    plt.show()
    return model

model_lasso = lasso()


# xgboost回归
def xgb_reg():
    model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=.1)  # 建立回归模型
    model_xgb.fit(X_train, Y_train)  # fit
    xgb_preds = np.expm1(model_xgb.predict(X_test))  # xgboost的预测结果
    lasso_preds = np.expm1(model_lasso.predict(X_test))  # lasso的预测结果
    predictions = pd.DataFrame({'xgboost': xgb_preds, 'lasso': lasso_preds})
    predictions.plot(x='xgboost', y='lasso', kind='scatter')  # 比较两者
    preds = 0.3*xgb_preds + 0.7*lasso_preds  # 融合
    return preds


# 结果提交
def submit():
    preds = xgb_reg()
    result = pd.DataFrame({'id': test['Id'], 'SalePrice': preds})
    result.to_csv('xgb_lasso.csv', index=False)

