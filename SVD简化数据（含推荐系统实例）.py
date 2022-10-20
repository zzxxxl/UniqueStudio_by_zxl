#day10.18：今天刚好看到SVD推荐系统，那就写一写吧
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#SVD:奇异值分解
#优点：简化数据，去除噪声，提高算法的结果
#缺点：数据的转化可能难以理解
#适用数据类型：数值型数据

#SVD可以看成是对PCA主特征向量的一种解法
#SVD的数据压缩：将一个大矩阵近似变成三个小矩阵，存储空间降低(不能简单地将降维和数据压缩理解为一个东西)

def loadexdata():
    return[[4,4,0,2,2],
           [4,0,0,3,3],
           [4,0,0,1,1],
           [1,1,1,2,0],
           [2,2,2,0,0],
           [1,1,1,0,0],
           [5,5,5,0,0]]

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
    
#SVD的应用之一就是信息检索，我们称利用SVD的方法为隐性语义索引(LSI)或隐性语义分析(LSA)
#另一应用：推荐系统简单版本的推荐系统能够计算项或者人之间的相似度，更先进的方法则先利用SVD从数据中构建一个主体空间,然后再在该空间下计算计算其相似度
#奇异值介绍：见PCA主成分分析算法
#保留奇异值数目的选择：如将奇异值的平方和累加到总值的90%为止；如保留前3000个
#基于协同过滤的推荐引擎；协同过滤：通过将用户和其他用户的数据进行对比来实现推荐
#当数据采用矩阵形式列出，就可以比较用户或物品之间的相似度；当知道两个用户或者两个物品之间的相似度，我们就可以利用已有的数据来预测位置的用户喜好

#相似度计算：基于用户对它们的意见来计算相似度
def ecludsim(ina,inb): #欧氏距离
    return 1.0 / (1.0 + np.linalg.norm(ina - inb)) #求二范数

def pearssim(ina,inb): #皮尔逊相关系数
    if len(ina) < 3:
        return 1.0 #完全相关
    else :
        return 0.5 + 0.5 * np.corrcoef(ina,inb,rowvar=0)[0][1]

def cossim(ina,inb): #余弦相似度
    num = float(ina.T * inb)
    denom = np.linalg.norm(ina) * np.linalg.norm(inb)
    return 0.5 + 0.5 * (num / denom)

#推荐引擎的评价：最小均方根误差
#推荐系统的工作：为用户返回N个最好的推荐菜
#基于物品相似度的推荐引擎

#计算用户对物品的估计评分值
def standest(datamat,user,simmeas,item): #数据矩阵、用户编号、相似度计算方法、物品编号;假设这里行对应用户，列对应物品
    n = np.shape(datamat)[1] #得到物品数目
    simtotal = 0.0; ratsimtotal = 0.0
    for j in range(n):#遍历物品
        userating =  datamat[user,j] #用户对物品j的评分
        if userating == 0:#如果用户对该物品没有评分就跳过
            continue
        overlap = np.nonzero(np.logical_and(datamat[:,item].A>0,\
                datamat[:,j].A>0))[0] #寻找目标物品与当前物品的相似度;logtical_and逻辑与函数
        #print(overlap)
        if len(overlap) == 0:#两个没有任何重合元素，则相似度为0
            similarity = 0
        else:
            similarity = simmeas(datamat[overlap,item],\
                    datamat[overlap,j])
            print("the %d and %d similarity is %f" %(item, j, similarity))     
            simtotal += similarity
            ratsimtotal += similarity * userating #相似度与当前用户评分的乘积
    if simtotal == 0:
        return 0
    else :
        return ratsimtotal/simtotal #对相似度评分的乘积进行归一化
    
#推荐引擎:产生最高的n个推荐结果
def recommend(datamat, user, n=3, simmeas=cossim,estmethod=standest):
    unrateditems = np.nonzero(datamat[user,:].A==0)[1] #寻找未评级的物品
    if len(unrateditems) == 0:
        return "you rated everything"
    itemscores = []
    for item in unrateditems:
        estimatedscore = estmethod(datamat,user,simmeas,item)
        itemscores.append((item,estimatedscore))
    return sorted(itemscores,key=lambda x:x[1],reverse=True)[:n] #寻找前n个未评级物品,key代表按照itemsscores中的第二个值进行从小到大排序,reverse=True代表翻转

#基于svd的评分估计
def svdest(datamat,user,sigmeas,item):
    n = np.shape(datamat)[1]
    simtotal = 0.0; ratesimtotal = 0.0
    U,sigma,VT = np.linalg.svd(datamat) #左奇异矩阵可以用于对行数的压缩，右奇异矩阵可以以用于对列得到压缩
    sig4 = np.mat(np.eye(4) * sigma[:4]) #构建对角矩阵
    #我不太理解这里的代码实现，不应该用datamat*datamat.T的svd值嘛
    xformeditems = datamat.T * U[:,:4] #* sig4.I  #构建转换后的物品(原书上这里加了一段，我觉得可以去掉)
    #print(xformeditems)
    for j in range(n):
        userrating = datamat[user,j]
        if (userrating == 0) or (j == item):
            continue
        similarity = sigmeas(xformeditems[item,:].T,\
                xformeditems[j,:].T)
        print("the %d and %d similarity is %f" %(item,j,similarity))
        simtotal += similarity
        ratesimtotal += similarity * userrating
    if simtotal == 0:
        return 0
    else :
        return ratesimtotal / simtotal

#基于SVD的图像压缩
def printmat(inmat,thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inmat[i,k] > thresh):
                print(1)
            else :
                print(0)
        print(' ')

#只需要两个奇异值就能对图像实现重构，总数字为130，与原来的1024相比实现几乎10倍的压缩比
def imgcompress(numsv=3,thresh=0.8):
    myl=[]
    for line in open("trainSet_PCA2.txt").readlines():
        newrow = []
        for i in range(32):
            newrow.append(int(line[i]))
        myl.append(newrow)
    mymat = np.mat(myl)
    print("original matrix")
    printmat(mymat,thresh)
    U,sigma,VT = np.linalg.svd(mymat)
    sigrecon = np.mat(np.zeros((numsv,numsv)))
    for k in range(numsv):
        sigrecon[k,k] = sigma[k]
    reconmat = U[:,:numsv] * sigrecon * VT[:numsv,:] #根据奇异值得到反构出的原矩阵
    print("reconstructed matrix using %d singular values" %numsv)
    printmat(reconmat,thresh)


if __name__ == '__main__':
    #mymat = np.mat(loadexdata())
    #print(recommend(mymat,2))
    #U,sigma,VT = np.linalg.svd(np.mat(loadExData2()))
    #print(sigma)
    #sig2 = sigma**2
    #print(sum(sig2))
    #print(sum(sig2[:3])) #前三个元素所包含的能量已经高于总能量的90%，于是可以将一个11维的矩阵转换成一个3维的矩阵
    #mymat = np.mat(loadExData2())
    #print(recommend(mymat,1,estmethod=svdest))
    imgcompress(2)