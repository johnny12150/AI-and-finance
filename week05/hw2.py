import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

S = 50
L = 40
T = 2
r = 0.08
vol = 0.2

def BLSprice(S, L, T, r, vol):
    d1 = (math.log(S/L)+(r+0.5*vol*vol)*T)/(vol*math.sqrt(T))
    d2 = d1- vol*math.sqrt(T)
    # call price的定價
    call = S*norm.cdf(d1)-L*math.exp(-r*T)*norm.cdf(d2)
    return call

def MCsim(S, T, r, vol, N):
    # delta t為時間差
    dt = T/ N
    St = np.zeros((N+1))
    St[0] = S
    for i in range(N):
        St[i+1] = St[i]*math.exp((r-0.5*vol*vol)*dt + np.random.normal()*vol*math.sqrt(dt))
    return St

# 1. M, N到多少會接近 black-scholes (用M, N)
N = 1000
M = 10000
call = 0
for i in range(M):
    Sa = MCsim(S, T, r, vol, N)
    # plt.plot(Sa)
    if Sa[-1] -L >0:
        # 等於0也就不用加了
        call += Sa[-1] -L

# 期望跟模擬的差異
print(abs(BLSprice(S, L, T, r, vol) - call/ M*math.exp(-r*T)))

# 2. 切的越多期會越接近black-scholes
# 測試不同N
N = 10000
def BTcall(S, T, R, vol, N, L):
    dt = T/ N
    u = math.exp(vol*math.sqrt(dt))
    d = math.exp(-vol*math.sqrt(dt))
    p = (math.exp(R*dt) - d)/ (u-d)
    priceT = np.zeros((N+1, N+1))
    priceT[0, 0] = S
    # N層樹
    for c in range(N):
        priceT[0][c+1] = priceT[0][c] * u
        for r in range(N):
            # 從右下開始推
            priceT[r+1][c+1] = priceT[r][c] * d  
    probT = np.zeros((N+1, N+1))
    probT[0][0] = 1
    for c in range(N):
        for r in range(N):
            # 往右推機率
            probT[r][c+1] += probT[r][c]*p
            # 往下推
            probT[r+1][c+1] += probT[r][c]*(1-p)
    call = 0
    for r in range(N+1):
        if priceT[r][N] >= L:
            call += (priceT[r][N] - L)*probT[r][N]
    return call*math.exp(-R*T)

print(abs(BLSprice(S, L, T, r, vol) - BTcall(S, T, r, vol, N, L)))

# 3. 拿今日選擇權價格，給定合理的S, r, T, L (不同L可以畫出一個微笑, 5口以上)
# S = 10889.96, r定存, T 10月第三週三日期, K 挑履約價, call成交
# 把推出的5個vol畫成圖

S = 10889.96
r = 0.008
T =  1/52 # 年化
# 挑五口
K = [10800, 10900, 11000, 11100, 11200]
call = [152, 82, 33, 9.2, 2.3]

def BisectionBLS(S, L, T, r, call, tol):
    left = 0.00000000000001
    right = 1
    while(right-left > tol):
        middle = (left+right)/2
        # 異號
        if (BLSprice(S, L, T, r, middle)-call) * (BLSprice(S, L, T, r, left)-call) < 0:
            right = middle
        else:
            left = middle
    return (left+right)/2

vol_list = []
# 算5口的vol
for i in range(5):
    vol_list.append(BisectionBLS(S, K[i], T, r, call[i], 0.00001))

plt.plot(vol_list)    
