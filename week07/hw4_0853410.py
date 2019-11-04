import numpy as np
import math
import matplotlib.pyplot as plt
import random
from sklearn import datasets
from collections import Counter


def kmeans(sample,K,maxiter):
    N = sample.shape[0]
    D = sample.shape[1]
    C = np.zeros((K,D))
    L = np.zeros((N,1))
    L1 = np.zeros((N,1))
    dist = np.zeros((N,K))
    idx = random.sample(range(N),K)
    C = sample[idx,:]
    iter = 0
    while(iter<maxiter):
        for i in range(K):
            dist[:,i] = np.sum((sample-np.tile(C[i,:],(N,1)))**2,1)
        L1 = np.argmin(dist,1)
        if(iter>0 and np.array_equal(L,L1)):
            break
        L = L1
        for i in range(K):
            idx = np.nonzero(L==i)[0]
            if(len(idx)>0):
                C[i,:] = np.mean(sample[idx,:],0)
        iter += 1
    wicd = np.sum(np.sqrt(np.sum((sample-C[L,:])**2,1)))
    return C,L,wicd


# load data
iris = datasets.load_iris()
feature =iris.data
target=iris.target

# 1. 原始資料
C,L,wicd = kmeans(feature,3,10000)

print(wicd)

# 2. standard score
def WICD(center, label, group):
    return np.sum(np.sqrt(np.sum((group-center[label,:])**2,1)))

G = feature
GA = (G-np.tile(np.mean(G,0),(G.shape[0],1)))/np.tile(np.std(G,0),(G.shape[0],1))

#  用kmeans算出的L跟原始的G比wicd
C2,L2,wicd2 = kmeans(GA,3,1000)
print(WICD(C2, L2, G))

# 3. scaling

# GC = (G-np.tile(np.min(G,0),(G.shape[0],1)))/(np.tile(np.max(G,0),(G.shape[0],1)) -np.tile(np.min(G,0),(G.shape[0],1)))
features = feature
GC = (features - np.tile(np.min(features,0),(features.shape[0],1)))/(np.tile(np.max(features,0),(features.shape[0],1))-np.tile(np.min(features,0),(features.shape[0],1)))
C3,L3,wicd3 = kmeans(GC,3,1000)
print(WICD(C3, L3, G))

# KNN
def knn(test,train, target,k):
    N = train.shape[0]
    dist = np.sum((np.tile(test, (N, 1)) - train)**2, 1)
    idx = sorted(range(len(dist)), key=lambda i: dist[i])[0:k]
    return Counter(target[idx]).most_common(1)[0][0]

for i in range(1, 11):
    confusion_matrix = np.zeros((3,3))
    for j in range(feature.shape[0]):
        # j 是當下的test
        train_idx = list(set(range(feature.shape[0])) - set([j]))
        predict = knn(feature[j], feature[train_idx, :], target, i)

        confusion_matrix[predict][target[j]]+=1
    print('N='+str(i))
    print(confusion_matrix)
