import numpy as np 
import pandas as pd  
import matplotlib.pyplot as plt 
import csv
import seaborn as sns  
import math
from sklearn.linear_model import LogisticRegression

def preprogressstrain():
    df_train = pd.read_csv("trainSet_tianic.csv")
    #观察发现年龄项、客舱以及embarked有缺失值(714non_null、204non_null和889non_null,其他数据都是891non_null)
    #print(df_train.describe())
    #df_train.info()
    #g = sns.heatmap(df_train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
    #plt.show()
    #g = sns.FacetGrid(df_train, col='Survived')
    #g.map(sns.histplot, "Age")
    #plt.show()

    #删除age和cabin(下策之选)
    #df_train.drop("Age",axis = 1,inplace = True)
    #df_train.drop("Cabin",axis = 1,inplace=True)
    #df_train.drop("Embarked",axis = 1,inplace = True)

    #如果删除数据有缺失的行，那么只剩183行，太多数据丢失，不合理
    #df_train.dropna(axis = 0,inplace = True)
    #df_train.info()

    #每个人的名字、车票、乘客编号都是不同的，无法对其进行训练，即使训练也无法用其进行预测，因此可以将其删除
    #而船舱数据确实太多且不具有训练意义，因此也选择删除
    df_train = df_train.drop(["Name","Ticket","PassengerId","Cabin"], axis = 1)
    #df_train.info()

    #性别用字符串进行标志不利于进行训练，因此这里使用map映射将男/女变成0/1,也可以使用get_dummies进行处理
    sex_map = {"male":0,"female":1}
    df_train["Sex"] = df_train["Sex"].map(sex_map)
    #print(df_train.head())
    #df_train = df_train.join(pd.get_dummies(df_train.Sex))
    #df_train.drop("Sex",axis = 1,inplace =True)
    #df_train.info()

    #对于embarked特征，观察其只有两个缺失值，因此尝试用众数进行代替
    #观察到”S“具有最多数量，用其对缺失值进行替代
    #print(df_train["Embarked"].value_counts())
    df_train["Embarked"] = df_train["Embarked"].fillna("S")
    #同理，对其进行字符值到数值的转换
    Embarked_map = {"S":0,"C":1,"Q":2}
    df_train["Embarked"] = df_train["Embarked"].map(Embarked_map)

    #对于年龄，考虑到其为连续性数据，尝试用已有年龄的平均值进行填补
    df_train["Age"] = df_train["Age"].fillna(df_train["Age"].mean())
    #fig = plt.figure()
    #plt.bar(df_train['Age'],df_train['Survived'])
    #plt.show()

    #考虑家庭情况
    df_train["Family_size"] = df_train["SibSp"] + df_train["Parch"] + 1
    df_train["Single"] = 1 #初始化每个人是一个人出行
    df_train.loc[df_train["Family_size"] == 1,"Single"] = 0 #如果有家人就不是单人辣
    #得到是否单人确实和生存率存在联系
    #print(df_train[["Single","Survived"]].groupby(["Single"],as_index=False).mean())
    #放弃Parch、SibSp、Family_size特征，保留Single
    df_train = df_train.drop(["Parch","SibSp","Family_size"],axis = 1)

    #考虑将年龄和票价这样的连续值分段变成简单的0,1,2...
    df_train.loc[df_train['Age'] <= 15,'Age'] = -2 #少年
    df_train.loc[(df_train['Age'] > 15) & (df_train['Age'] <= 30),'Age'] = -1 #青年 这里要用&且要打括号不然会报错
    df_train.loc[(df_train['Age'] > 30) & (df_train['Age'] <= 45),'Age'] = 0 #壮年
    df_train.loc[(df_train['Age'] > 45) & (df_train['Age'] <= 60),'Age'] = 1 #中年
    df_train.loc[df_train['Age'] > 60,'Age'] = 2 #老年

    #这里对费用采用独热编码，也可以使用均值归一化
    df_train.loc[df_train['Fare'] <= 7.91,'Fare'] = -2
    df_train.loc[(df_train["Fare"] > 7.91) & (df_train["Fare"] <= 14.454),'Fare'] = -1
    df_train.loc[(df_train["Fare"] > 14.454) & (df_train["Fare"] <= 31),'Fare'] = 1
    df_train.loc[df_train["Fare"] > 31,'Fare'] = 2
    #df_train["Fare"] = normalize(df_train["Fare"])
    # min_value = df_train['Fare'].min(0)
    # max_value = df_train['Fare'].max(0)
    # rangevalue = max_value - min_value
    # m = np.shape(df_train)[0]
    # normal_data = df_train["Fare"] - np.tile(min_value,(m,1))
    # normal_data = normal_data / np.tile(range,(m,1))
    # df_train["Fare"] = normal_data
    #for i in range(m):
        #df_train["Fare"][i] = (df_train["Fare"][i] - min_value)/rangevalue
    #print(df_train.head())
    #print(np.tile(min_value,(m,1)))
    #normdataset = np.zeros((m,1))
    #normdataset = df_train["Fare"] - np.tile(min_value,(m,1)) #tile为拓展功能
    #normdataset = normdataset/np.tile(rangevalue,(m,1))

    #尝试用KNN进行填补(我觉得KNN填补应该在其他数据都处理好之后，因为这样方便计算距离)
    #df_train["Age"].fillna(0,inplace=True) #这里先填0方便计算机距离
    #print(df_train.head())
    #df_train.info()
    return df_train

def normalize(data):
    min_value = data.min(0)  # 每一列最小值
    max_value = data.max(0)
    ranges = max_value - min_value
    normal_matrix = data - np.tile(min_value, (data.shape[0], 1))
    normal_matrix = normal_matrix / np.tile(ranges, (data.shape[0], 1))
    return normal_matrix

def sigmoid(inx):
    return 1.0/(1+np.exp(-inx))

        
def loaddataset():
    xdata = []
    ydata = []
    #我想把数据换成向量形式，找了好久，终于成功了呜呜呜
    with open("Changedtianic.csv",'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        data = []
        for line in reader:
            data.append(line)
        #print(header)
        #print(data)
    for line in range(len(data)):
        ydata.append(int(data[line][1]))
        xarr = [1.0]
        for j in range(2,len(data[line])):
            xarr.append(float(data[line][j]))
        xdata.append(xarr)
    #print(xdata)
    return xdata,ydata #数据终于处理好辣
    
#批量梯度下降
def gradascent(datamat,classlabels):
    datamatrix = np.mat(datamat) #特征数组变成矩阵
    labelmat = np.mat(classlabels).transpose() #变成y的列矩阵 m×1
    m,n = np.shape(datamatrix) #得到特征矩阵的行，列
    alpha = 0.001 #学习率
    maxcycles = 4500
    ceita = np.ones((n,1)) #n行1列的单位矩阵
    #print(datamatrix)
    #print(ceita)
    for k in range(maxcycles):
        h = sigmoid(datamatrix * ceita) #m×n 乘以 n×1 得到m×1矩阵这里是点乘
        error = h - labelmat
        ceita = ceita - alpha * datamatrix.T * error#这里写成datamatrix.transpose()效果一样
    return ceita


#引入正则化参数的梯度下降
def granscent_with_matrix(datamat,classlabels):
    datamatrix = np.mat(datamat)
    labelmat = np.mat(classlabels).T
    m,n = np.shape(datamatrix)
    alpha = 0.001
    maxcycles = 4500
    numberda =  0.01 #设置正则化参数
    ceita = np.ones((n,1))
    for k in range(maxcycles):
        h = sigmoid(datamatrix * ceita)
        ceita_up = ceita.copy()
        ceita_up[0] = 0
        error =  datamatrix.T * (h - labelmat)  + numberda * ceita_up
        ceita = ceita -  alpha * error 
    return ceita


def preprogresstest():
    df_test = pd.read_csv("testSet_tianic.csv")
    #test数据缺少的是age（332），费用（417），客舱（91）
   # df_test.info()
    #我们按照类似训练集的处理
    df_test["Age"]  = df_test["Age"].fillna(df_test["Age"].mean())
    #df_test = df_test.drop('Cabin',axis = 1)
    df_test.drop(["Name","Ticket","PassengerId","Cabin"],axis = 1,inplace = True)
    map_sex = {'male':0,'female':1}
    df_test["Sex"] = df_test["Sex"].map(map_sex)
    map_embarked = {"S":0,"C":1,"Q":2}
    df_test["Embarked"] = df_test["Embarked"].map(map_embarked)
    df_test["Family_size"] = df_test["SibSp"] + df_test["Parch"] + 1
    df_test["Single"] = 1 
    df_test.loc[df_test["Family_size"] == 1,"Single"] = 0 
    df_test = df_test.drop(["Parch","SibSp","Family_size"],axis = 1)
    df_test.loc[df_test['Age'] <= 15,'Age'] = -2
    df_test.loc[(df_test['Age'] > 15) & (df_test['Age'] <= 30),'Age'] = -1
    df_test.loc[(df_test['Age'] > 30) & (df_test['Age'] <= 45),'Age'] = 0
    df_test.loc[(df_test['Age'] > 45) & (df_test['Age'] <= 60),'Age'] = 1 
    df_test.loc[df_test['Age'] > 60,'Age'] = 2 
    df_test["Fare"].fillna(df_test["Fare"].median(),inplace = True)
    df_test.loc[df_test['Fare'] <= 7.91,'Fare'] = -2
    df_test.loc[(df_test["Fare"] > 7.91) & (df_test["Fare"] <= 14.454),'Fare'] = -1
    df_test.loc[(df_test["Fare"] > 14.454) & (df_test["Fare"] <= 31),'Fare'] = 1
    df_test.loc[df_test["Fare"] > 31,'Fare'] = 2
    return df_test
    #print(df_test.info())

def testdata(xarr,yarr,ceita):
    xarr = np.array(xarr)
    yarr = np.array(yarr)
    yhat = []
    for i in range(len(xarr)):
        yhat.append(sigmoid(int(np.sum(xarr[i,:] * ceita))))
    #print(yhat)
    yy = [1 if yhat[i] > 0.5 else 0 for i in range(len(yhat))]
    #print(yy)
    numerror = 0
    for i in range(len(yarr)):
        if yy[i] != yarr[i]:
            numerror += 1
    print(float(numerror/len(yarr))*100)
    return float(numerror/len(yarr))*100

#随机梯度下降(为一种在线学习算法，速度更快)
#波动地收敛，接近全局最小值(够用啦)
#没有矩阵的转置过程过程中使用的数据类型都是numpy数组
def stocgradscent0(datamatrix,classlabels):
    m,n = np.shape(datamatrix)
    alpha = 0.07
    ceita = np.ones(n) #ceita为1行n列的行向量
    for i in range(m):
        h = sigmoid(sum(datamatrix[i]*ceita)) #这里得到的是一个数值,datamatrix[i]*ceita是一个向量（感觉有点广播的意思在里面），sum函数求向量和
        #print(datamatrix[i])
        #print(ceita)
        #print(datamatrix[i] * ceita) #这里的乘不是点乘，datamatrix[i]和ceita维度相同(都是1×n)，因此这里是对用位置数值相乘
        error = h - classlabels[i] 
        ceita = ceita - alpha * error * datamatrix[i]
    return ceita

#随机梯度下降的改进
def stocgradscent1(datamatrix, classlabels, numiter = 150):
    m,n = np.shape(datamatrix)
    ceita = np.ones(n)
    for j in range(numiter):#迭代次数
        dataindex = list(range(m))
        for i in range(m):
            alpha = 2 / (1.0 + j + i) + 0.0001 #alpha(学习率)在每次迭代时都会进行调整，可以缓解数据波动或高频波动，另外，虽然alpha会随着迭代次数不断减小，但永远不会减小到0，因为还有常数项，这样可以保证在多次迭代后新数据仍然具有一定的影响
            #当j<<max(i)时，alpha就不是严格下降的，避免参数的严格下降也常见于模拟退火等其他优化算法中
            randinidex = int(np.random.uniform(0,len(dataindex))) #通过选取随机数来更新回归系数，能减少周期性的波动;生成下界为0，上界为len的int型随机数
            h = sigmoid(sum(datamatrix[randinidex]*ceita))
            error = h - classlabels[randinidex]
            ceita = ceita - alpha * error * datamatrix[randinidex]
            del(dataindex[randinidex])
    return ceita

#adam优化算法
def adam(xdata,ydata,alpha,iterations):
    m,dim = np.shape(xdata)
    theta = np.zeros(dim) #参数
    momentum = 0.01 #冲量
    threshold = 0.0001 #停止迭代的错误阈值
    error = 0 #初始错误为0

    b1 = 0.9 #默认值
    b2 = 0.999 #默认值
    e = 0.000000001 #默认值
    mt = np.zeros(dim)
    vt = np.zeros(dim)

    for i in range(iterations):
        j = i % m
        error = 1 / (2 * m) * np.dot((np.dot(xdata,theta) - ydata).T,
        (np.dot(xdata,theta) - ydata))
        if abs(error) <= threshold:
            break
        gradient = xdata[j] * (np.dot(xdata[j],theta) - ydata[j])
        mt = b1 * mt + (1 - b1) * gradient
        vt = b2 * vt + (1 - b2) * (gradient**2)
        mtt = mt / (1 - (b1**(i + 1)))
        vtt = vt / (1 - (b2**(i + 1)))
        vtt_sqrt = np.array([math.sqrt(vtt[i])for i in range(len(vtt))])
        theta = theta - alpha * mtt / (vtt_sqrt + e)
    return theta

#利用KNN算法填补缺失数据
#思路：查找k个距离最近的数据，取其平均值
#这里之前用0填补方便查找
def KNN_workon_lost(dataset,labels,k):
    #print(dataset)
    white = []
    for i in range(len(dataset)):
        if dataset[i][2] != 0:
            continue
        else :
            white.append(i)
    print(white)
    #for i in range(white):
    dataset = np.array(dataset)
    dataset = np.column_stack((dataset,labels))
    #数据终于处理好咧，接下来进入正题啦
    for i in range(len(white)):
        datasize = dataset.shape[0] #返回dataset的行数
        diffmat = np.tile(dataset[i],(datasize,1))-dataset #先将inx向右复制一遍（相当于没变），再向下复制datasize遍，并减去dataset得到距离
        sqdiffmat = diffmat**2 #距离平方
        sqdistances = sqdiffmat.sum(axis=1) #与每个点的总距离（合并每一列也就是每个点）
        distances = sqdistances**0.5 #距离开根号
        sorteddistances = distances.argsort() #按distance中的元素从小到大排序，并提取对应的索引输出
        distancesum = 0.0
        for i in range(k): #取前k个最近的索引
            distancesum += sorteddistances[i]
        dataset[i][2] = distancesum/k
    dataset = np.delete(dataset,6,axis = 1)
    print(dataset[:,2])

def loaddataset1():
    xdata = []
    #我想把数据换成向量形式，找了好久，终于成功了呜呜呜
    with open("Changedtianic1.csv",'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        data = []
        for line in reader:
            data.append(line)
        #print(header)
        #print(data)
    for line in range(len(data)):
        xarr = [1.0]
        for j in range(1,len(data[line])):
            xarr.append(float(data[line][j]))
        xdata.append(xarr)
    #print(xdata)
    return xdata#数据终于处理好辣

def answerdata(xarr,ceita):
    xarr = np.array(xarr)
    yhat = []
    for i in range(len(xarr)):
        yhat.append(sigmoid(int(np.sum(xarr[i,:] * ceita))))
    #print(yhat)
    yy = [1 if yhat[i] > 0.5 else 0 for i in range(len(yhat))]
 
    with open("Tianic_submission.csv","w",newline='') as csvfile: # 1. 创建文件对象
        writer = csv.writer(csvfile)   #2. 基于文件对象构建 csv写入对象
        writer.writerow(["PassengerId","Survived"])
        for i in range(len(yy)):
            writer.writerow([892+i,int(yy[i])])
        csvfile.close()       # 4. 关闭文件



if __name__=="__main__":
    train = preprogressstrain()
    train.to_csv("Changedtianic.csv")
    xdata,ydata = loaddataset()
    #尝试用KNN进行填补(我觉得KNN填补应该在其他数据都处理好之后，因为这样方便计算距离)
    #KNN_workon_lost(xdata,ydata,10)
    #print(xdata)
    #print(ydata)
    #数据切割
    xdata=np.array(xdata)
    ydata=np.array(ydata)
    xdata1=xdata[:600]
    ydata1=ydata[:600]
    xdata2=xdata[600:]
    ydata2=ydata[600:]
    #ceita = gradascent(np.array(xdata),np.array(ydata))
    #ceita = stocgradscent0(np.array(xdata),ydata)
    ceita = stocgradscent1(np.array(xdata),ydata,200)
    #ceita = adam(np.array(xdata1),np.array(ydata1),0.03,5000)#0.03 5000 22.68
    #ceita = granscent_with_matrix(np.array(xdata),np.array(ydata))
    ##罪恶，尝试调用sklearn
    #log_reg = LogisticRegression()
    #log_reg.fit(xdata1,ydata1)
    # predicts = log_reg.predict(xdata2)
    # numerror = 0
    # for i in range(len(ydata2)):
    #     if predicts[i]!= ydata2[i]:
    #         numerror += 1
    # print(float((numerror)/len(ydata2))*100)

    #print(ceita)
    #test = preprogresstest()
    #print(test)
    #test = np.array(test)
    #yhat = []
    #for i in range(len(test)):
        #yhat.append(sigmoid(int(sum(test[i,:] * ceita))))
    #print(yhat)
    #yy = [1 if yhat[i] > 0.5 else 0 for i in range(len(yhat))]
    #print(yy)
    errorrate = testdata(xdata2,ydata2,ceita)
    #print(xdata)
    #print(ydata)
    test = preprogresstest()
    test.to_csv("Changedtianic1.csv")
    xtest = loaddataset1()
    answerdata(xtest,ceita)


    