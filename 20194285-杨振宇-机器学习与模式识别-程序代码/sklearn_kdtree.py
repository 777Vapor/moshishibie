from sklearn.neighbors import KDTree
import pandas as pd
import numpy as np
import time
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def get_hog_features(trainset):
    features = []

    hog = cv2.HOGDescriptor('../hog.xml')

    for img in trainset:
        img = np.reshape(img,(28,28))
        cv_img = img.astype(np.uint8)

        hog_feature = hog.compute(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features.append(hog_feature)

    features = np.array(features)
    features = np.reshape(features,(-1,324))

    return features
def KNN_KDthree(testset,trainset,train_labels,test_labels):
    predict = []
    count = 0
    tree = KDTree(trainset)

    for i in range(0, len(test_labels)):
        print(i)
        test_vec = testset[[i]]
        # 输出当前运行的测试用例坐标，用于测试
        print(count)
        count += 1
        knn_list = []
        dist_to_knn, indx_knn = tree.query(test_vec, k=k_value)
        for i in range(0,5):
           labels = train_labels[indx_knn[0][i]]
           knn_list.append(labels)
        # 统计选票
        class_total = 10
        class_count = [0 for i in range(class_total)]
        for train_label in knn_list:
            class_count[train_label] += 1
        # 找出最大选票
        max_vote = max(class_count)
        # 找出最大选票标签
        for i in range(class_total):
            if max_vote == class_count[i]:
                predict.append(i)
                break


    return np.array(predict)

if __name__ == '__main__':
    k_value=5
    time_1 = time.time()
    raw_data = pd.read_csv('../data/mnist_784.csv',header=0)
    data = raw_data.values
    character = data[0::, 1::]
    labels = data[::,0]

    # 选取训练集数量
    hog = get_hog_features(character)
    x_train, x_test, y_train, y_test = train_test_split(hog, labels, train_size=60000, test_size=10000, random_state=10)

    final_test = KNN_KDthree(x_test, x_train, y_train,y_test)
    final_score = accuracy_score(y_test, final_test)
    print("测试集上准确率为 ", final_score)
    time_3 = time.time()
    print('测试集上总共用时 ', time_3 - time_1, ' second', '\n')