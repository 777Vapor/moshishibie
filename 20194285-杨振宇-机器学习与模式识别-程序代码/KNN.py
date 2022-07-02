#20194285 杨振宇自动化1904
#数据载入已经将28*28的数据格式转化为CSV文件784*1的格式，便于进行距离计算
#此程序参考了李航的《统计学习方法》，p37的内容
import decorator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import statistics
from sklearn.model_selection import KFold

def KNN(testset,trainset,train_labels):

    predict = []
    count = 0

    for test_vec in testset:
        # 输出当前运行的测试用例坐标，用于测试
        print (count)
        count += 1
        knn_list = []       # 当前k个最近邻居
        max_index = -1      # 当前k个最近邻居中距离最远点的坐标
        max_dist = 0        # 当前k个最近邻居中距离最远点的距离
        # 先将前k个点放入k个最近邻居中，填充满knn_list
        for i in range(k):
            label = train_labels[i]
            train_vec = trainset[i]
            dist = np.linalg.norm(train_vec - test_vec)         # 计算两个点的欧氏距离
            knn_list.append((dist,label))
        # 剩下的点
        for i in range(k,len(train_labels)):
            label = train_labels[i]
            train_vec = trainset[i]
            dist = np.linalg.norm(train_vec - test_vec)         # 计算两个点的欧氏距离
            # 寻找k个邻近点钟距离最远的点
            if max_index < 0:
                for j in range(k):
                    if max_dist < knn_list[j][0]:
                        max_index = j
                        max_dist = knn_list[max_index][0]

            # 如果当前k个最近邻居中存在点距离比当前点距离远，则替换
            if dist < max_dist:
                knn_list[max_index] = (dist,label)
                max_index = -1
                max_dist = 0

        # 统计选票
        print(knn_list)
        class_total = 10
        class_count = [0 for i in range(class_total)]
        for dist,label in knn_list:
            class_count[label] += 1

        # 找出最大选票
        max_vote = max(class_count)

        # 找出最大选票标签
        for i in range(class_total):
            if max_vote == class_count[i]:
                predict.append(i)
                break

    return np.array(predict)


if __name__ == '__main__':
    best_k, best_score = 0, 0
    kf = KFold(n_splits=5, random_state=None)
    #数组a、c、d分别用来储存k值为i时的五折交叉验证平均准确率，召回率和精确度
    #数组b储存这一轮次的k值i
    a = []
    b = []
    c=[]
    d=[]
    print('Start read data')
    #读取csv格式的数据
    time_1 = time.time()
    raw_data = pd.read_csv('../data/mnist_784.csv',header=0)
    data = raw_data.values
    character = data[0::, 1::]
    labels = data[::,0]

    # 选取训练集数量
    x_train, x_test, y_train, y_test = train_test_split(character, labels, train_size=60000, test_size=10000, random_state=10)

   # 将k从5到20逐个进行五折交叉运算以寻找合适的k
    k_range = range(5, 21)
    for k in k_range:
        kfoldscore = []
        kfoldrecall = []
        kfoldpre=[]
        # 数组kfoldscore = [],kfoldrecall = [],kfoldpre=[]分别用来储存k值为i时的五折交叉验证每一折的准确率，召回率和精确度，用于计算平均值
        # 显示5折交叉验证的具体划分情况
        for train_index, test_index in kf.split(x_train,y_train):
            print("Train:", train_index, "Validation:", test_index)
            X_train, X_test = x_train[train_index], x_train[test_index]
            Y_train, Y_test = y_train[train_index], y_train[test_index]

        i = 1
        for train_index, test_index in kf.split(x_train,y_train):
            print('\n{} of kfold {}'.format(i, kf.n_splits))
            X_train, X_test = x_train[train_index], x_train[test_index]
            Y_train, Y_test = y_train[train_index], y_train[test_index]
            pred_test = KNN(X_test, X_train, Y_train)
            score = accuracy_score(Y_test, pred_test)
            recall = recall_score(Y_test, pred_test,average='macro')
            precision = precision_score(Y_test, pred_test,average='macro')
            kfoldscore.append(score)
            kfoldrecall.append(recall)
            kfoldpre.append(precision)
            print("此折的准确率为 %f" % score)
            print("此折的召回率为 %f" % recall)
            print("此折的为精确度 %f" % precision)
            i += 1

        time_2 = time.time()
        # 输出准确率，kfoldscore是储存每一折准确率的数组，everage是这一轮的平均准确率
        average = statistics.mean(kfoldscore)
        averagerecall = statistics.mean(kfoldrecall)
        averagepre = statistics.mean(kfoldpre)
        a.append(average)
        b.append(k)
        c.append(averagerecall)
        d.append(averagepre)
        print('训练用时 ', time_2 - time_1, ' second', '\n')
        print("当k等于 ", k)
        print("平均准确率为%f"% average)
        print("平均召回率为%f" % averagerecall)
        print("平均精确度为%f" % averagepre)

        #最优解求取
        if average > best_score:
         best_k, best_score = k, average
    time_3 = time.time()
    print('训练总共用时 ', time_3 - time_1, ' second', '\n')

    #输出最优解
    print("最优k:", best_k)  #
    print("最优k的测试集准确率:", best_score)
    #ab数组用于储存k值和准确率
    print(a)
    print(b)
    #绘制图像
    plt.plot(b, a)
    plt.plot(b, c)
    plt.plot(b, d)
    plt.legend(['accuracy_score', 'recall_score', 'precision_score'], loc='upper right')
    plt.show()
    '''此步骤用于测试，删掉此行和最后一行即可运行
    #测试集运算
    k = 5
    final_test = KNN(x_test, x_train, y_train)
    final_score = accuracy_score(y_test, final_test)
    print("测试集上准确率为 ", final_score)
    time_3 = time.time()
    print('测试集上总共用时 ', time_3 - time_1, ' second', '\n')
   
'''
