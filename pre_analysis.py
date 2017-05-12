#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : pre_analysis.py
# @Time    : 2017-05-09 14:59
# @Author  : zhang bo
# @Note    : 使用高级回归的思想来预测房子价格
# @Describe: 数据的详细说明参考  https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
"""
# 导入包
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
import warnings
import xgboost as xgb

warnings.filterwarnings('ignore')

'''数据预处理'''
# 加载训练集和测试集
path_train = 'D:\PythonProjects\HousePrices\data\\train.csv'
path_test = 'D:\PythonProjects\HousePrices\data\\test.csv'
df_train = pd.read_csv(path_train)
df_train.columns

# SalePrice
df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice'])
print('Skewness: %f' % df_train['SalePrice'].skew())  # 偏斜度：1.882876
print('Kurtosis: %F' % df_train['SalePrice'].kurt())  # 峰值：6.536282

# GrLivArea:线性正相关
data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)  # 合并两个列
data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000))  # 绘制散点图

# TotalBsmtSF：线性
data = pd.concat([df_train['SalePrice'], df_train['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0, 800000))

# 分析categorical features：OverallQual  相关
data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)
fig = sns.boxplot(x='OverallQual', y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)

# YearBuilt: 用户更倾向于购买新建的房子
data = pd.concat([df_train['SalePrice'], df_train['YearBuilt']], axis=1)
fig = sns.boxplot(x='YearBuilt', y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)  # 旋转90度


'''特征选择'''
corrmat = df_train.corr()  # 相关性矩阵
fig = sns.heatmap(corrmat, vmax=.8, square=True)  # 相似度热度图
plt.xticks(rotation=90)

k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index  # 与SalePrice相关性最高的10个特征
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True,fmt='.2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values)
plt.xticks(rotation=90)

# 去除冗余的特征，及放弃相似度较高的特征
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size=2.5)

'''缺失值处理'''
missing_total = df_train.isnull().sum().sort_values(ascending=False)  # 统计缺失值的总数
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)  # 计算每个特征的缺失率
missing_data = pd.concat([missing_total, percent], axis=1, keys=['Total', 'Percent'])  # 合并
missing_data.head()
# 通过分析，所有具有缺失值的特征，都可以丢弃，除了Electrical
df_train = df_train.drop(missing_data[missing_data['Total'] > 1].index, axis=1)  # 丢弃缺失个数大于1的特征
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)  # 删除Electrical 有缺失的记录
df_train.isnull().sum().max()  # 检验是否还有缺失值的存在

'''单变量分析'''
# 确定上下界
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:, np.newaxis])  # 标准化
range_index = saleprice_scaled[:, 0].argsort()  # 排序后返回index
low_range = saleprice_scaled[range_index][:10]    # saleprice最小的10个
high_range = saleprice_scaled[range_index][-10:]  # saleprice最大的10个

'''双变量分析'''
data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000))
# GrLivArea异常点分析
df_train.sort_values(by='GrLivArea', ascending=False)[:2]  # 离群点 id=1299， 524
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
# TotalBsmtSF 异常点分析
data = pd.concat([df_train['SalePrice'], df_train['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0, 800000))

'''数据挖掘'''
# 以下主要考虑几个性质：Normality、Homoscedasticity、Linearity、Absence of correlated errors

'''1.正态性检验'''
sns.distplot(df_train['SalePrice'], fit=norm)  # SalePrice
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)  # 如果拟合较好，说明趋于正态分布
# 由于SalePrice不是正态分布，使用log函数可以改善
df_train['SalePrice'] = np.log(df_train['SalePrice'])
sns.distplot(df_train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)

# 同理，检验GrLivArea
sns.distplot(df_train['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)

df_train['GrLivArea'] = np.log(df_train['GrLivArea'])  # log矫正
sns.distplot(df_train['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)

# 检验TotalBsmtSF
sns.distplot(df_train['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
# 由于该特征有0值，所以不能直接进行log，需要将0和非0的加以区分
df_train['BsmtSF'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)  # 新建一列，记录非0
df_train['BsmtSF'] = 0
df_train.loc[df_train['TotalBsmtSF'] > 1, 'BsmtSF'] = 1  # 非0标记为1
df_train.loc[df_train['BsmtSF'] == 1, 'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])  # 将非0的log化
# 将非0的绘图展示
sns.distplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit=norm)
fig = plt.figure()
reg = stats.probplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot=plt)

'''2. Homoscedasticity'''
plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])
df_train = pd.get_dummies(df_train)
df_train = df_train.fillna(df_train.mean())


'''modeling'''
X_test = pd.read_csv('D:\PythonProjects\HousePrices\data\\test.csv')
X_train = df_train.drop('SalePrice', axis=1)
Y_train = df_train['SalePrice']
X_test = pd.get_dummies(X_test)
X_test = X_test.fillna(X_test.mean())


