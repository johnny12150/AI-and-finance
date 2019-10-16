import math
import pandas as pd
import numpy as np

p1 = 10/ 14
p2 = 4/ 14

entropy1 = -p1*math.log2(p1)- p2*math.log2(p2)

# 正跟負
def entropy(p1, n1):
    if p1 == 0 and n1 == 0:
        return 1
    elif p1 == 0:
        return 0
    elif n1 == 0:
        return 0
    pp = p1/(p1+n1)
    pn = n1/(p1+n1)
    return -pp*math.log2(pp)-pn*math.log2(pn)


# information gain
def IG(p1, n1, p2, n2):
    num1 = p1+n1
    num2 = p2+n2
    num = num1 + num2
    return entropy(p1+p2,n1+n2)-num1/num*entropy(p1,n1)-num2/num*entropy(p2,n2)

# IG(9, 1, 1, 3)

# pd.read_csv('./PlayTennis.txt', sep=" ", header=None)
data = np.loadtxt('./PlayTennis.txt', usecols=range(5), dtype='int')
feature = data[:, 0:4]
target = data[:, 4] -1
node = dict()
node['data'] = range(len(target))
Tree = []
Tree.append(node)
# 處理到的node數
t = 0
# 樹會越切越長
while t < len(Tree):
    idx = Tree[t]['data']
    # 判斷是不是leaf node
    if sum(target[idx]) == 0:
        # 都不打球
        Tree[t]['leaf'] = 1
        Tree[t]['decision'] = 0
    elif sum(target[idx]) == len(idx):
        Tree[t]['leaf'] = 1
        Tree[t]['decision'] = 1
    else:
        bestIG = 0
        for i in range(feature.shape[1]):
            # 第 i clo的特徵取集合後做排序
            pool = list(set(feature[idx, i]))
            pool.sort() 
            for j in range(len(pool)-1):
                thres = (pool[j]+pool[j+1])/2
                G1 = []
                G2 = []
                for k in idx:
                    if feature[k, i] <= thres:
                        G1.append(k)
                    else:
                        G2.append(k)
                # 計算1的個數
                thisIG = IG(sum(target[G1] == 1), sum(target[G1] == 0), sum(target[G2] ==1 ), sum(target[G2] ==0 ))
                if thisIG > bestIG:
                    bestIG = thisIG
                    bestG1 = G1
                    bestG2 = G2
                    bestthres = thres
                    bestf = i

        if bestIG >0:
            Tree[t]['leaf'] = 0
            Tree[t]['selectf'] = bestf
            Tree[t]['threshold'] = bestthres
            Tree[t]['child'] = [len(Tree), len(Tree)+1]
            node = dict()
            node['data'] = bestG1
            Tree.append(node)
            node = dict()
            node['data'] = bestG2
            Tree.append(node)
        else:
            Tree[t]['leaf'] = 1
            if sum(target[idx]==1) > sum(target[idx]==0):
                Tree[t]['decision'] = 1
            else:
                Tree[t]['decision'] = 0

    t+=1

for i in range(len(target)):
    test_features = feature[i, :]
    now =0
    while Tree[now]['leaf'] ==0:
        if test_features[Tree[now]['selectf']]<= Tree[now]['threshold']:
            now = Tree[now]['child'][0]
        else:
            now = Tree[now]['child'][1]

    print(target[i], Tree[now]['decision'])

