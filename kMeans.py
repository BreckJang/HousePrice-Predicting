# coding=utf-8
import os
from os import path
import numpy as np
from numpy import *
import pandas as pd
#import saxtest
from matplotlib import pyplot as plt

# 计算两个向量的欧氏距离
def compute_dist(vect_a, vect_b):
    return sqrt(sum(power(vect_a - vect_b, 2)))

os.chdir('./data') #current path


# 随机创建k个质心
def rand_cent(data_set, k):

    n = shape(data_set)[1]  # 列数
    centroids = np.mat(zeros((k, n)))  # k*n的矩阵
    for j in range(1, n):  # 对矩阵的每一列
        min_j = min(data_set[:, j])  # 每一列的最小值
        range_j = float(max(data_set[:, j]) - min_j)  # 每列的极差，即为了保证质心必须位于数据边界之内
        centroids[:, j] = min_j + range_j * random.rand(k, 1)  # 列向量
    return centroids


# k-means算法实现
def k_means(data_set, k, song_id, outf1):
    data_set = mat(data_set)
    m = shape(data_set)[0] 
    cluster_assment = mat(zeros((m, 2)))
    # 1. 创建初始质心
    centroids = rand_cent(data_set, k)
    # 2.质心改变，计算距离
    cluser_changed = True
    sse_set = []
    while cluser_changed:
        cluser_changed = False
        for i in range(m):  # 对于每个样本
            min_dist = inf  # 无穷大
            min_index = -1
            for j in range(k):  # 遍历每个质心，寻找与样本i最近的质心
                # 计算每个质心分别与每个样本之间的距离
                dist_ji = compute_dist(centroids[j, :], data_set[i, :])
                if dist_ji < min_dist:
                    min_dist = dist_ji
                    min_index = j  # 确定最近的质心点
            if cluster_assment[i, 0] != min_index:
                cluser_changed = True  # 质心改变
            cluster_assment[i, :] = min_index, min_dist ** 2
    cluster_assment2=pd.DataFrame(cluster_assment)
    song_cluser = pd.DataFrame([song_id,cluster_assment2.ix[:,0],cluster_assment2.ix[:,1]]).T
    song_cluser.to_csv(outf1,header = None)
 
       # 3.更新质心的值
"""
        for cent in range(k):  # 对于每个质心
           pst_in_cluster = data_set[nonzero(cluster_assment[:, 0].A == cent)[0]]  # 过滤获得所有点
           centroids[cent, :] = mean(pst_in_cluster, axis=0)  # 更新质心--求所有点的均值
          
   # return centroids, cluster_assment

# 聚类结果展示
def show(data_set, k):
    centroids, cluster_assment = k_means(data_set, k)
    numSamples, dim = data_set.shape
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in xrange(numSamples):
        mark_index = int(cluster_assment[i, 0])
        plt.plot(data_set[i, 0], data_set[i, 1], mark[mark_index])
        mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
        for i in range(k):
            plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
    plt.show()
"""


# 主函数入口
if __name__ == '__main__':

    path = 'part2_guiyi_song_dayly_play_3.csv'
    
    of = pd.read_csv(path,header=None)
    song_id= of.ix[:,0]
    f = of.drop([0],axis=1)
    k = 5
    #data_set = mat(load_dataset(path))
    #data_set = mat(load_csv(path2))
    
    k_means(f, k,song_id,'part2_user_actions.csv')
    #plt.show()

    #show(data_set, k)