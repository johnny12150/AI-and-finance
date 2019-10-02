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
b1 = -putr(11100) + putr(10900)
b2 = -putr(11000) + putr(10900)
b3 = callr(10900) - callr(11000)
b4 = callr(10900) - callr(11100)
b6 = callr(11000) - callr(11100)
b7 = -putr(11100) + putr(11000)
# 買權
plt.plot(x, b3, x, b4, x, b6)
# 賣權
plt.plot(x, b1, x, b2, x, b7)

# 第二小題
# A. straddle
straddle1 = callr(11000) + putr(11000)
straddle2 = -callr(11000) - putr(11000)
plt.plot(x, straddle1, 'r', x, straddle2, 'r')

# B. strangle
st1 = callr(11200) + putr(10800)
st2 = -callr(11200) - putr(10800)
plt.plot(x, st1, 'g', x, st2, 'g')

# 比較
plt.plot(x, straddle1, 'r', x, st1, 'g')
plt.plot(x, straddle2, 'r', x, st2, 'g')

# 第三小題
bu1 = putr(10800) + putr(11200) - putr(11000)*2
bu2 = putr(10700) + putr(11300) - putr(11000)*2
plt.plot(x, bu1, x, bu2)

