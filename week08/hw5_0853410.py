import numpy as np
import math
import matplotlib.pyplot as plt

tprice = np.load('./data.npy')

def E(tc,b,w,py):
    return np.sum(abs(priceABC(b,w,py,tc)-tpricelog[:tc]))/tc

# 求A, B, C
def priceABC(b,w,py,tc):
    A4 = np.zeros((tc,3))
    pprice=np.zeros(tc)
    for i in range(tc):
        temp = (tc-i)**b
        A4[i,0] = 1
        A4[i,1] = temp
        A4[i,2] = temp*np.cos(w*math.log(tc-i)+py)
    x4 = np.linalg.lstsq(A4,tpricelog[:tc])[0]
    for i in range(tc):
        pprice[i]=x4[0] + x4[1]*((tc-i)**b) + x4[2]*((tc-i)**b)*math.cos(w*math.log(tc-i)+py)
    return pprice

tpricelog=np.zeros(len(tprice))
for i in range(len(tpricelog)):
    tpricelog[i]=math.log(tprice[i])

def price(A,B,C,b,w,py,tc,t):
    return A+B*((tc-t)**b)+C*((tc-t)**b)*math.cos(w*math.log(tc-t)+py)

# 在1155-1166發生bubble

peo=10000
goodpeo=100
pop = np.random.randint(0,2,(peo,20))
fit = np.zeros((peo))
for generation in range(10):
    print("generation:",generation)
    for i in range(peo):
        gene = pop[i,:]
        A = np.sum(2**np.array(range(7))*gene[0:7])//4+1155              #tc
        B = (np.sum(2**np.array(range(7))*gene[7:14]))/128               #b
        C = np.sum(2**np.array(range(3))*gene[14:17])*2                  #w
        D = (np.sum(2**np.array(range(3))*gene[17:20]))/8*2*math.pi      #fi
        fit[i]=E(A,B,C,D)
    sortf = np.argsort(fit)
    pop = pop[sortf,:]
    for i in range(goodpeo,peo):
        fid = np.random.randint(0,goodpeo)
        mid = np.random.randint(0,goodpeo)
        while(mid==fid):
            mid = np.random.randint(0,goodpeo)
        mask = np.random.randint(0,2,(1,20))
        son = pop[mid,:]
        father = pop[fid,:]
        son[mask[0,:]==1]=father[mask[0,:]==1]
        pop[i,:] = son
    for i in range(goodpeo):
        m = np.random.randint(0,peo)
        # 用20個基因描述
        n = np.random.randint(0,20)
        if(pop[m,n]==0):
            pop[m,n]=1
        else:
            pop[m,n]=0


for i in range(peo):
    gene = pop[i,:]
    A = np.sum(2**np.array(range(7))*gene[0:7])//4+1155               #tc, 4bit, 1155-1166
    B = (np.sum(2**np.array(range(7))*gene[7:14]))/128               #b
    C = np.sum(2**np.array(range(3))*gene[14:17])*2                  #w
    D = (np.sum(2**np.array(range(3))*gene[17:20]))/8*2*math.pi      #fi
    fit[i]=E(A,B,C,D)
sortf = np.argsort(fit)
pop = pop[sortf,:]

bestgene = pop[0,:]
bestTc = np.sum(2**np.array(range(7))*gene[0:7])//4+1155           #tc
bestBeta = (np.sum(2**np.array(range(7))*gene[7:14]))/128         #b
bestW = np.sum(2**np.array(range(3))*gene[14:17])*2               #w
bestfi = (np.sum(2**np.array(range(3))*gene[17:20]))/8*2*math.pi  #fi


A4 = np.zeros((bestTc,3))
for i in range(bestTc):
    temp = (bestTc-i)**bestBeta
    A4[i,0] = 1
    A4[i,1] = temp
    A4[i,2] = temp*np.cos(bestW*math.log(bestTc-i)+bestfi)
x4 = np.linalg.lstsq(A4,tpricelog[:bestTc])[0]

x = np.arange(0,bestTc)
y = np.zeros((bestTc))
for i in range(bestTc):
    y[i] = math.exp(price(x4[0],x4[1],x4[2],bestBeta,bestW,bestfi,bestTc,i))


plt.plot(tprice)
plt.plot(x,y)
