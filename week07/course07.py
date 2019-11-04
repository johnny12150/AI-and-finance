import numpy as np
import math
import matplotlib.pyplot as plt
import random
import collections

def kmeans(sample,K,maxiter):
    N = sample.shape[0]
    D = sample.shape[1]
    C = np.zeros((K,D))  # K個中心點/K群的座標
    L = np.zeros((N,1))  # label
    L1 = np.zeros((N,1))
    dist = np.zeros((N,K))
    idx = random.sample(range(N),K)
    C = sample[idx,:]
    iter = 0
    while(iter<maxiter):
        # K個中心點分開算
        for i in range(K):
            # np.tile會repeat, 每個中心點repeat N*1次
            dist[:,i] = np.sum((sample-np.tile(C[i,:],(N,1)))**2, axis=1)
        L1 = np.argmin(dist,1)
        if(iter>0 and np.array_equal(L,L1)):
            print(iter)
            break
        L = L1
        for i in range(K):
            # 取非0的index
            idx = np.nonzero(L==i)[0]
            if(len(idx)>0):
                C[i,:] = np.mean(sample[idx,:],0)
        iter += 1
    # 每次更新做圖
    # G1 = G[L==0,:]
    # G2 = G[L==1,:]
    # G3 = G[L==2,:]
    # plt.plot(G1[:,0],G1[:,1],'r.',G2[:,0],G2[:,1],'g.',G3[:,0],G3[:,1],'b.',C[:,0],C[:,1],'kx') 
    
    # C[L,:] -> K類各自center
    wicd = np.sum(np.sqrt(np.sum((sample-C[L,:])**2,1)))
    return C,L,wicd


# 模擬資料
G1 = np.random.normal(0,1,(5000,2))  # 5000筆mean=0, 標準差=1
G1 += 4
# G1[:,0] = G1[:,0] + 4
# G1[:,1] = G1[:,1] + 4

# 畫G1
# plt.plot(G1[:,0], G1[:,1], '.')

G2 = np.random.normal(0,1,(3000,2))
G2[:,1] = G2[:,1]*3 - 3
G = np.append(G1,G2,axis=0)
# 畫G
# plt.plot(G[:,0], G[:,1], '.')

G3 = np.random.normal(0,1,(2000,2))
G3[:,1] = G3[:,1]*4
c45 = math.cos(-45/180*math.pi)
s45 = math.sin(-45/180*math.pi)
R = np.array([[c45,-s45],[s45,c45]])
G3 = G3.dot(R)
G3[:,0] = G3[:,0]-4
G3[:,1] = G3[:,1]+6
G = np.append(G,G3,axis=0)
# plt.plot(G[:,0],G[:,1],'.')

C,L,wicd = kmeans(G,3,1000)
G1 = G[L==0,:]
G2 = G[L==1,:]
G3 = G[L==2,:]
plt.plot(G1[:,0],G1[:,1],'r.',G2[:,0],G2[:,1],'g.',G3[:,0],G3[:,1],'b.',C[:,0],C[:,1],'kx')
print(C,L,wicd)

# 標準化
GA = (G-np.tile(np.mean(G,0), (G.shape[0], 1))/ np.tile(np.std(G, 0), (G.shape[0], 1)))

def knn(test,train, target,k):
    N = train.shape[0]
    dist = np.sum((np.tile(test, (N, 1)) - train)**2, 1)
    idx = sorted(range(len(dist)), key=lambda i: dist[i])[0:k]
    return Counter(target[idx]).most_common(1)[0][0]
