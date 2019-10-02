import numpy as np
import matplotlib.pyplot as plt

# 買權價格
call = {10700:266, 10800:189, 10900:120, 11000:67, 11100:32, 11200:13, 11300:6.1}
# 賣權價格
put = {10700:43, 10800:63, 10900:93, 11000:139, 11100:207, 11200:290, 11300:360}
x = np.arange(10500, 11501)

def callr(K):
    global x
    global call
    # 賺/ 賠多少
    return np.maximum(x-K, 0) - call[K]

def putr(K):
    global x
    global put
    # 賺/ 賠多少
    return np.maximum(K-x, 0) - put[K]

# 第一小題
b1 = callr(10900) - putr(11000) # X
b2 = callr(10900) - putr(11100) # X
b3 = callr(10900) - callr(11000)
b4 = callr(10900) - callr(11100)
b5 = callr(11000) - putr(11100) # X
b6 = callr(11000) - callr(11100)
b7 = -putr(11100) + putr(11000)
# 剩餘兩種put - put
plt.plot(x, b7, 'y', [x[0], x[-1]], [0, 0], '--')

# 第二小題
# A. straddle
straddle1 = callr(11100) + putr(11100)
straddle2 = -callr(11100) - putr(11100)
plt.plot(x, straddle1, 'r', x, straddle2, 'r')

# B. strangle
st1 = callr(11200) + putr(10800)
st2 = -callr(11200) - putr(10800)
plt.plot(x, st1, 'g', x, st2, 'g')

# 第三小題
bu1 = putr(10800) + putr(11200) - putr(11000)*2
bu2 = -putr(10800) - putr(11200) + putr(11000)*2
plt.plot(x, bu1, x, bu2)

