import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

def MCsim(S, T, r, vol, N):
    # delta t為時間差
    dt = T/ N
    St = np.zeros((N+1))
    St[0] = S
    for i in range(N):
        St[i+1] = St[i]*math.exp((r-0.5*vol*vol)*dt + np.random.normal()*vol*math.sqrt(dt))
    return St

def BLSprice(S, L, T, r, vol):
    d1 = (math.log(S/L)+(r+0.5*vol*vol)*T)/(vol*math.sqrt(T))
    d2 = d1- vol*math.sqrt(T)
    # call price的定價
    call = S*norm.cdf(d1)-L*math.exp(-r*T)*norm.cdf(d2)
    return call


S =50
L =40
T =2
r =0.08
vol = 0.2
N = 100

M =100000
call =0
for i in range(M):
    Sa = MCsim(S, T, r, vol, N)
    # plt.plot(Sa)
    if Sa[-1] -L >0:
        # 等於0也就不用加了
        call += Sa[-1] -L

print(call/ M*math.exp(-r*T))

# TODO: 幾次and幾期接近 black-shoes
call_e = BLSprice(S, L, T, r, vol)
print(abs(call_e - call/ M*math.exp(-r*T)))

# 二元樹
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

# TODO: 切的越多期會越接近black-shoes
print(BTcall(S, T, r, vol, N, L))



