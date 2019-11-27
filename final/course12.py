import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def show(profit):
    profit2 = np.cumsum(profit)
    plt.plot(profit2)
    plt.show()
    ans1 = profit2[-1]
    ans2 = np.sum(profit >0)/ len(profit)
    ans3 = np.mean(profit[profit > 0])
    ans4 = np.mean(profit[profit<=0])
    plt.hist(profit, bins=100)
    plt.show()
    print('Total: ', ans1, '\nWin ration: ', ans2, '\nWin avg: ', ans3, '\nLose avg: ', ans4)


df = pd.read_csv('./final/data/TXF20112015_org.csv', sep=',', header=None)
df3 = pd.read_csv('./final/data/TXF20112015.csv', sep=',', header=None)
df.rename(columns=df.iloc[0], inplace=True)
df.drop(0, inplace=True)
df.reset_index(drop=True, inplace=True)
df = df.drop('dCode', axis=1)
df2 = df.iloc[:, 1:].astype(int)
# TAIEX = df.values
TAIEX = df3.values
# TAIEX[:, 1:] = df2.values
# TAIEX[:, 0] = np.array([int(w.strip()) if type(w) == str else int(w) for w in TAIEX[:, 0].tolist()])
tradeday = list(set(TAIEX[:, 0]//10000))
tradeday.sort()

# 策略0.0 開盤買收盤賣
profit = np.zeros((len(tradeday), 1))
for i in range(len(tradeday)):
    date = tradeday[i]

    idx = np.nonzero(TAIEX[:, 0]//10000 == date)[0]
    idx.sort()
    profit[i] = TAIEX[idx[-1], 1] - TAIEX[idx[0], 2]

show(profit)

# 策略0.1
profit = np.zeros((len(tradeday), 1))
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:, 0]//10000 == date)[0]
    idx.sort()
    profit[i] = TAIEX[idx[0], 2] - TAIEX[idx[-1], 1]

show(profit)

profit = []
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:, 0] // 10000 == date)[0]
    idx.sort()
    p1 = TAIEX[idx[0], 2]
    idx2 = np.nonzero(TAIEX[idx, 4]<=p1-30)[0]
    if len(idx2) == 0:
        p2 = TAIEX[idx[-1], 1]
    else:
        p2 = TAIEX[idx[idx2[0]], 1]
    profit.append(p2-p1)

show(profit)

# 策略2.0
profit = []
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:, 0] // 10000 == date)[0]
    idx.sort()
    p1 = TAIEX[idx[0], 2]
    idx2 = np.nonzero(TAIEX[idx, 4]<=p1-30)[0]
    idx3 = np.nonzero(TAIEX[idx, 3] >= p1 + 30)[0]
    if len(idx2) == 0 and len(idx3)==0:
        p2 = TAIEX[idx[-1], 1]
    elif len(idx3)==0:
        p2 = TAIEX[idx[idx2[0]], 1]
    elif len(idx2) == 0:
        p2 = TAIEX[idx[idx3[0]], 1]
    elif idx2[0] < idx3[0]:
        p2 = TAIEX[idx[idx2[0]], 1]
    else:
        p2 = TAIEX[idx[idx3[0]], 1]
    profit.append(p2-p1)

show(np.array(profit))

# todo: 實作到策略3.1
