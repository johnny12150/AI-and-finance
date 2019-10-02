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

y1 = callr(10900)

def putr(K):
    global x
    global put
    # 賺/ 賠多少
    return np.maximum(K-x, 0) - put[K]

y2 = putr(10900)
# 買方的獲利曲線
plt.plot(x, y1, 'r',x, y2, 'g', [x[0], x[-1]], [0, 0], '--')

# 賣方獲利曲線
y3 = -putr(10900)
plt.plot(x, y3, 'b', [x[0], x[-1]], [0, 0], '--')

# 組合
y4 = y1 + y3
plt.plot(x, y1, 'r',x, y3, 'g', x, y4, 'y', [x[0], x[-1]], [0, 0], '--')

# bull spread wth two put
y5 = putr(10700)
y6 = -putr(11100)
y7 = y5 + y6
plt.plot(x, y5, 'r',x, y6, 'g', x, y7, 'y', [x[0], x[-1]], [0, 0], '--')

# bull spread差距
b1 = putr(10700) - putr(11100)
b2 = putr(10700) - putr(10900)
plt.plot(x, b1, 'r',x, b2, 'g', [x[0], x[-1]], [0, 0], '--')

# straddle
s1 = callr(10900) + putr(10900)
s2 = -callr(10900) - putr(10900)
plt.plot(x, callr(10900), 'r',x, putr(10900), 'g', x, s1, 'y', [x[0], x[-1]], [0, 0], '--')
plt.plot(x, -callr(10900), 'r',x, -putr(10900), 'g', x, s2, 'y', [x[0], x[-1]], [0, 0], '--')

# strangle (價高的買權, 價低的賣權)
st1 = callr(10900) + putr(10700)
plt.plot(x, callr(10900), 'r',x, putr(10700), 'g', x, st1, 'y', [x[0], x[-1]], [0, 0], '--')

# conversion 買一口買權, 賣一口賣權
c1 = callr(10900) - putr(10900)
plt.plot(x, callr(10900), 'r',x, -putr(10900), 'g', x, c1, 'y', [x[0], x[-1]], [0, 0], '--')

# butterfly spread 買低高的賣權, 賣中的賣權
bu1 = putr(10800) + putr(11200) - putr(11000)*2
plt.plot(x, putr(10800), 'r',x, putr(11200), 'g', x, - putr(11000)*2, 'orange', x, bu1, 'o', [x[0], x[-1]], [0, 0], '--')

