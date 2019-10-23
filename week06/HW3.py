import math
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
np.random.seed(38)
idx = np.random.permutation(150)
feature_org = iris.data[idx, :]
target_org=iris.target[idx]

def entropy(p1,n1):
    if(p1==0 and n1==0):
        return 1
    value = 0
    pp = p1/(p1+n1)
    pn = n1/(p1+n1)
    if(pp>0):
        value -= pp*math.log2(pp)
    if(pn>0):
        value -= pn*math.log2(pn)
    return value

def IG(p1,n1,p2,n2):
    num = p1+n1+p2+n2
    num1 = p1+n1
    num2 = p2+n2
    return entropy(p1+p2,n1+n2)-num1/num*entropy(p1,n1)-num2/num*entropy(p2,n2)


def tree1(target,feature,s):
    node=dict()
    node['data']=range(len(target))
    Tree = []
    Tree.append(node)
    t = 0
    while(t<len(Tree)):
        idx = Tree[t]['data']
        if(sum(target[idx])==0):
            Tree[t]['leaf']=1
            Tree[t]['decision']=0
        elif(sum(target[idx])==len(idx)):
            Tree[t]['leaf']=1
            Tree[t]['decision']=1
        else:
            bestIG = 0
            for i in range(feature.shape[1]):
                pool = list(set(feature[idx,i]))
                pool.sort()
                for j in range(len(pool)-1):
                    thres = (pool[j]+pool[j+1])/2
                    G1 = []
                    G2 = []
                    for k in idx:
                        if(feature[k,i]<=thres):
                            G1.append(k)
                        else:
                            G2.append(k)
                    thisIG = IG(sum(target[G1]==0),sum(target[G1]==1),sum(target[G2]==0),sum(target[G2]==1))
                    if(thisIG>bestIG):
                        bestIG = thisIG
                        bestG1 = G1
                        bestG2 = G2
                        bestthres = thres
                        bestf = i
            if(bestIG>0):
                Tree[t]['leaf']=0
                Tree[t]['selectf']=bestf
                Tree[t]['threshold']=bestthres
                Tree[t]['child']=[len(Tree),len(Tree)+1]
                node = dict()
                node['data']=bestG1
                Tree.append(node)
                node = dict()
                node['data']=bestG2
                Tree.append(node)
            else:
                Tree[t]['leaf']=1
                if(sum(target[idx]==1)>sum(target[idx]==0)):
                    Tree[t]['decision']=1
                else:
                    Tree[t]['decision']=0
        t+=1

    test_feature = iris['data'][s,:]
    now = 0
    while(Tree[now]['leaf']==0):
        bestf = Tree[now]['selectf']
        thres = Tree[now]['threshold']
        if(test_feature[bestf]<=thres):
            now = Tree[now]['child'][0]
        else:
            now = Tree[now]['child'][1]
    return (Tree[now]['decision'])  
    
def tree2(target,feature,s):
    node=dict()
    node['data']=range(len(target))
    Tree = []
    Tree.append(node)
    t = 0
    while(t<len(Tree)):
        idx = Tree[t]['data']
        if(sum(target[idx])==len(idx)):
            Tree[t]['leaf']=1
            Tree[t]['decision']=1
        elif(sum(target[idx])==2*len(idx)):
            Tree[t]['leaf']=1
            Tree[t]['decision']=2
        else:
            bestIG = 0
            for i in range(feature.shape[1]):
                pool = list(set(feature[idx,i]))
                pool.sort()
                for j in range(len(pool)-1):
                    thres = (pool[j]+pool[j+1])/2
                    G1 = []
                    G2 = []
                    for k in idx:
                        if(feature[k,i]<=thres):
                            G1.append(k)
                        else:
                            G2.append(k)
                    thisIG = IG(sum(target[G1]==1),sum(target[G1]==2),sum(target[G2]==1),sum(target[G2]==2))
                    if(thisIG>bestIG):
                        bestIG = thisIG
                        bestG1 = G1
                        bestG2 = G2
                        bestthres = thres
                        bestf = i
            if(bestIG>0):
                Tree[t]['leaf']=0
                Tree[t]['selectf']=bestf
                Tree[t]['threshold']=bestthres
                Tree[t]['child']=[len(Tree),len(Tree)+1]
                node = dict()
                node['data']=bestG1
                Tree.append(node)
                node = dict()
                node['data']=bestG2
                Tree.append(node)
            else:
                Tree[t]['leaf']=1
                if(sum(target[idx]==1)>sum(target[idx]==2)):
                    Tree[t]['decision']=1
                else:
                    Tree[t]['decision']=2
        t+=1
    test_feature = iris['data'][s,:]
    now = 0
    while(Tree[now]['leaf']==0):
        bestf = Tree[now]['selectf']
        thres = Tree[now]['threshold']
        if(test_feature[bestf]<=thres):
            now = Tree[now]['child'][0]
        else:
            now = Tree[now]['child'][1]
    return (Tree[now]['decision'])  

def tree3(target,feature,s):
    node=dict()

    node['data']=range(len(target))
    Tree = []
    Tree.append(node)
    t = 0
    while(t<len(Tree)):
        idx = Tree[t]['data']
        if(sum(target[idx])==0):
            Tree[t]['leaf']=1
            Tree[t]['decision']=0
        elif(sum(target[idx])==2*len(idx)):
            Tree[t]['leaf']=1
            Tree[t]['decision']=2
        else:
            bestIG = 0
            for i in range(feature.shape[1]):
                pool = list(set(feature[idx,i]))
                pool.sort()
                for j in range(len(pool)-1):
                    thres = (pool[j]+pool[j+1])/2
                    G1 = []
                    G2 = []
                    for k in idx:
                        if(feature[k,i]<=thres):
                            G1.append(k)
                        else:
                            G2.append(k)
                    thisIG = IG(sum(target[G1]==0),sum(target[G1]==2),sum(target[G2]==0),sum(target[G2]==2))
                    if(thisIG>bestIG):
                        bestIG = thisIG
                        bestG1 = G1
                        bestG2 = G2
                        bestthres = thres
                        bestf = i
            if(bestIG>0):
                Tree[t]['leaf']=0
                Tree[t]['selectf']=bestf
                Tree[t]['threshold']=bestthres
                Tree[t]['child']=[len(Tree),len(Tree)+1]
                node = dict()
                node['data']=bestG1
                Tree.append(node)
                node = dict()
                node['data']=bestG2
                Tree.append(node)
            else:
                Tree[t]['leaf']=1
                if(sum(target[idx]==0)>sum(target[idx]==2)):
                    Tree[t]['decision']=0
                else:
                    Tree[t]['decision']=2
        t+=1
    test_feature = iris['data'][s,:]
    now = 0
    while(Tree[now]['leaf']==0):
        bestf = Tree[now]['selectf']
        thres = Tree[now]['threshold']
        if(test_feature[bestf]<=thres):
            now = Tree[now]['child'][0]
        else:
            now = Tree[now]['child'][1]
    return (Tree[now]['decision'])

confusion_matrix = np.zeros((3,3))
count=0
# cross-validate
# for k in range(5, 0, -1):
for k in range(5):
    for Q in range(30):
        # 從後面開始當validate
        # Q = idx[Q+(k-1)*30]
        Q = idx[Q+k*30]

        train_idx=[]
        target1 = []
        train_idx2=[]
        target2 = []
        train_idx3=[]
        target3 = []
        
        # 各類別的idx
        for i in range(len(target_org)):
            # if i<k*30 and i>= (k-1)*30:
            if i>= k*30 and i< (k+1)*30:
                continue
            if(target_org[i]==0):
                train_idx.append(i)
                target1.append(0)
            elif(target_org[i]==1):
                train_idx.append(i)
                target1.append(1)	
                
            if(target_org[i]==1):
                train_idx2.append(i)
                target2.append(1)
            elif(target_org[i]==2):
                train_idx2.append(i)
                target2.append(2)		
                
            if(target_org[i]==0):
                train_idx3.append(i)
                target3.append(0)
            elif(target_org[i]==2):
                train_idx3.append(i)
                target3.append(2)		

        feature = feature_org[train_idx,:]
        feature2 = feature_org[train_idx2,:]
        feature3 = feature_org[train_idx3,:]
        
        target11=np.array(target1)
        target22=np.array(target2)
        target33=np.array(target3)
        
        # modeling
        a=tree1(target11,feature,Q)
        b=tree2(target22,feature2,Q)
        c=tree3(target33,feature3,Q)

        CC=[]
        CC.append(a)
        CC.append(b)
        CC.append(c)
        # 三棵樹的各自投票
        count0=CC.count(0)
        count1=CC.count(1)
        count2=CC.count(2)
        
        if count0 == count1 and count1 == count2:
            X=2
        elif count0>=count1 and count0>=count2:
            X=0
        elif count1>=count0 and count1>=count2:
            X=1
        elif count2>=count0 and count2>=count1:
            X=2
        else:
            X=2
        
        # 正確
        if X == iris.target[Q]:
            confusion_matrix[X][X] += 1

        # 錯誤個數
        if X!=iris.target[Q]:
            count +=1
            confusion_matrix[X][iris.target[Q]] += 1

# 正確率
print((150 - count)/ 150)
