"""
day10.17:今天是周一，写个PCA降维算法
对数据进行简化的原因：使得数据更易采集；降低很多算法的计算开销；去除噪声；是的结果易懂
PCA（主成分分析）算法：数据由原来的坐标系转换到了新的坐标系，新坐标的选择是有数据本身决定的
第一个新坐标轴选择的是原始数据中方差最大的方向（覆盖数据最大差异性的坐标轴），第二个新坐标轴选择和第一个坐标轴正交且具有最大方差的方向(覆盖数据次大差异性的坐标轴)。该过程一直重复，重复次数为原始数据种特征的数目
大部分方差都包含在最前面的几个新坐标系中;通过数据集的协方差矩阵及特征值分析，我们就可以求得这些主成分的值，一旦得到了协方差矩阵的特征数量，我们就可以保留最大的N个值，这些特征向量也给出了N个最重要的特征的真实结构
因子分析算法：假设在观察数据的生成中由一些观察不到的隐变量，假设观察数据是这些隐变量和某些噪声的线性组合，那么隐变量的数据可能比观察数据的数目少，也就是说通过找到隐变量就可以实现数据的降维，因子分析已经用于社会科学、金融和其他领域中了
独立成分分析算法（ICA）：ICA假设数据是从N哥数据源生成的，这一点和因子分析有些类似，假设数据为多个数据按的混合观察结果，这些数据源之间在统计上是相互独立的，而在PCA中只假设数据是不相关的，同因子分析一样，如果数据源的数目少于观察的数目，则可以实现降维过程
PCA的优点：降低数据的复杂性，识别最重要的多个特征；缺点：不一定需要，且可能损失有用信息；使用数据类型：数值型数据
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def loaddataset(filename,delim='\t'):
    fr = open(filename)
    stringarr = [line.strip().split(delim) for line in fr.readlines()]
    dataarr = [list(map(float,line)) for line in stringarr]
    return np.mat(dataarr)


#原理分析：为什么要用什么协方差矩阵，为什么取前k个最大的，为什么样本在“协方差矩阵C的最大K个特征值所对应的特征向量”上的投影就是k维理想特征？
#其中一种解释是: 最大方差理论:方差越大，信息量就越大。协方差矩阵的每一个特征向量就是一个投影面
#每一个特征向量所对应的特征值就是原始特征投影到这个投影面之后的方差。由于投影过去之后，我们要尽可能保证信息不丢失
#所以要选择具有较大方差的投影面对原始特征进行投影，也就是选择具有较大特征值的特征向量
#然后将原始特征投影在这些特征向量上，投影后的值就是新的特征值。每一个投影面生成一个新的特征，k个投影面就生成k个新特征。
#PCA降维的目的，就是为了在尽量保证“信息量不丢失”的情况下，对原始特征进行降维，也就是尽可能将原始特征往具有最大信息量的维度上进行投影。将原特征投影到这些维度上，使降维后信息量损失最小。
#这里说的是真好呀
#原理分析： 那为什么协方差矩阵的特征向量可以看做是投影面，相对应的特征值是原始特征投影到这个投影面之后的方差？
#见csdn：https://blog.csdn.net/lanyuelvyun/article/details/82384179?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166601628016782391870413%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166601628016782391870413&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-2-82384179-null-null.142^v58^js_top,201^v3^control&utm_term=pca&spm=1018.2226.3001.4187
#原理分析：为什么计算协方差矩阵之前，要将样本的原始特征x1,x2...进行去均值化操作？
#见上链接，减去均值（中心化）这一步，是由PCA定义的所规定的必须步骤
def pca(datamat,topnfeat=9999999):#第二个参数为应用的特征个数，如果不指定它的值，那么函数就会返回前9999999个特征或者原始数据中全部的特征
    meanvals = np.mean(datamat,axis=0)#对各列求均值，得到m×1的矩阵
    meanremoved = datamat - meanvals #减去原始数据集的平均值(去均值化)
    covmat = np.cov(meanremoved,rowvar=0) #计算协方差(衡量两个变量的总体误差)矩阵 cov(x,y) = ((求和)(xi-x(均))(yi-y(均)))/n-1;rowvar=0代表此时每一行代表一个观测，每一列代表一个变量（属性）这里得到的x*x.T，所以下一行求得的特征向量就是U
    eigvals,eigvects = np.linalg.eig(np.mat(covmat)) #求矩阵的特征值和特征向量(其实这个特征值开根号就是svd函数的第二个返回值(奇异值)，吴恩达老师的机器学习课程就是利用这个参数降维的)(这里特征值和特征向量一一对应，我们要找到最大的topnfeat个特征值对应的特征向量)
    show_cov(eigvals,datamat)
    eigvalind = np.argsort(eigvals) #argsort(x)是将x中的元素从小到大排列(argsort(-x)则是从大到小)，得到的是其对应的索引index
    eigvalind = eigvalind[:-(topnfeat):-1] #根据特征值排序结果的逆序就可以得到topnfeat个最大的特征向量 [遍历起点：遍历终点：步长]
    redeigvects = eigvects[:,eigvalind]#将特征值最大的topnfeat个特征向量取出来，组成压缩矩阵
    lowddatamat = meanremoved * redeigvects #将原始数据转换到新空间，使得维度下降
    reconmat = (lowddatamat * redeigvects.T) + meanvals #利用降维后的矩阵访欧出元数据矩阵(用作测试，可与未压缩的矩阵比对)
    return lowddatamat,reconmat

"""
关于奇异值的解释
对于方阵而言A=Q*K*Q-1次方,其中的K就是特征向量。但是对于不是方阵的矩阵而言就没有特征向量。
非方阵的矩阵可以用奇异值分解来描述这个矩阵。A=U*K*V的转置。其中U叫做左奇异值，叫做奇异值，V的转置叫做右奇异值。因为K只有对角线的数不为0，并且数值是从大到小排列，所以一般只取r个，r的值越接近A的列数，那么三个矩阵的乘法得到的矩阵越接近A。
因为三个矩阵的面积之和远远小于原矩阵A，所以当我们向压缩空间表达A的时候，可以使用这三个矩阵。
当A不是矩阵的时候，把A转置变为A的转置。并且(A*A的转置)*v=pv。其中的v就是右奇异值。q=根号p,这里的p就是上面的奇异值。u=A*v/q，这里的u就是上面的左奇异值。
A为m×n矩阵，u为m×m，k为m×n，v为n×n,其中k是对矩阵A的奇异值分解，k除了对角线元素不为0，其他元素都为0并且对角元素从大到小排列，s中有n个奇异值，一般排在后面的比较接近0，所以仅保留比较大的r个奇异值（以一维矩阵的形式返回，且后面有0会舍去）
关于特征值和特征向量：
设A为n阶方阵，如果存在数m和非零n维列向量，使得Ax=mx成立，那么称m是A的额一个特征值，x为A的对应于m的特征向量
奇异值分解与特征值分解的关系；
M=UKV*,V*是V的共轭转置
M*M=VK*U*UKV*=V(K*K)V*
MM*=UKV*VK*U*=U(KK*)U*
V的列向量(右奇异向量)是M*M的特征向量
U的列向量(左奇异矩阵)是MM*的特征向量
K的非零对角元(非零奇异值)是MM*或M*M的非零特征值的平方根
奇异值分解在统计中的主要应用为PCA，数据集的特征值(在SVD中用奇异值表示)按照重要性排列，降维的过程就是舍弃不重要的特征向量的过程，而剩下的特征向量张成空间为降维后的空间
"""


def pictureshow(datamat,reconmat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datamat[:,0].flatten().A[0],datamat[:,1].flatten().A[0],marker='^',s=90) #flatten将数组折叠成一个一维的数据（这里默认按横的方向降，.A[0]是将矩阵变成数组）
    ax.scatter(reconmat[:,0].flatten().A[0],reconmat[:,1].flatten().A[0],marker='o',s=50,c='red')
    plt.show()

def replace_nan_with_mean():#将nan替换成平均值
    datamat = loaddataset("trainSet_PCA.data",' ') #这里是.data文件
    #print(datamat)
    numfeat = np.shape(datamat)[1]
    for i in range(numfeat):
        meanval = np.mean(datamat[np.nonzero(~np.isnan(datamat[:,i].A))[0],i])#对于每列非nan位置求平均 #isnan给定每个位置的True/False，nonzero得到两个描述非0位置的数组，取[0]即得到描述fei行的，即对第i列非nan的行值求平均
        datamat[np.nonzero(np.isnan(datamat[:,i].A))[0],i] = meanval
    #print(datamat)
    return datamat

def show_cov(featvalue,datamat):
    featvalue = sorted(featvalue)[::-1]
    # 同样的数据绘制散点图和折线图,这里拿前20个点绘图
    plt.scatter(range(1, 20), featvalue[:19])
    plt.plot(range(1, 20), featvalue[:19])
    
    # 显示图的标题和xy轴的名字
    # 最好使用英文，中文可能乱码
    plt.title("Scree Plot")  
    plt.xlabel("Factors")
    plt.ylabel("Eigenvalue")
    
    plt.grid()  # 显示网格
    plt.show()  # 显示图形

#SVD:奇异值分解
#优点：简化数据，去除噪声，提高算法的结果
#缺点：数据的转化可能难以理解
#适用数据类型：数值型数据


#SVD的应用之一就是信息检索，我们称利用SVD的方法为隐性语义索引(LSI)或隐性语义分析(LSA)
#另一应用：推荐系统简单版本的推荐系统能够计算项或者人之间的相似度，更先进的方法则先利用SVD从数据中构建一个主体空间,然后再在该空间下计算计算其相似度
#奇异值介绍：见上
#基于协同过滤额推荐引擎

    

if __name__ == '__main__':
    #datamat = loaddataset("testSetPCA.txt")
    #lowdmat,reconmat = pca(datamat,2)
    #pictureshow(datamat,reconmat)
    #可以观察到，前6个主成分覆盖率数据96.8%的方差，而前20个主成分覆盖了99.3%的方差，表明如果保留前6个而取出后584个主成分，就可以实现大概100：1的压缩比
    #另外，由于舍弃了噪声的主成分，将后面的主成分取出便使得数据更加干净
    datamat = replace_nan_with_mean()
    lowdmat,reconmat = pca(datamat,6)
    #pictureshow(datamat,reconmat)
    