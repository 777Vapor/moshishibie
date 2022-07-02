# 自动化1904杨振宇20194285
#此程序参考了李航的《统计学习方法》，p63到p65的内容
import cv2
import time
import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold

# 二值化
def binaryzation(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV, cv_img)
    return cv_img

def binaryzation_features(trainset):
    features = []
#将784*1的数据转化为28*28的数据并二值化
    for img in trainset:
        img = np.reshape(img, (28, 28))
        cv_img = img.astype(np.uint8)
        img_b = binaryzation(cv_img)
        features.append(img_b)

    features = np.array(features)
    features = np.reshape(features, (-1, feature_len))

    return features


class Tree(object):
    def __init__(self, node_type, Class=None, feature=None):
        self.node_type = node_type  # 节点类型为叶子节点或内部节点
        self.dict = {}  # dict的键表示特征Ag的可能值ai，值表示根据ai得到的子树
        self.Class = Class  # 叶节点表示的类，若是内部节点则为none
        self.feature = feature  # 表示当前的树即将由第feature个特征划分（即第feature特征是使得当前树中信息增益最大的特征）

    def add_tree(self, key, tree):
        self.dict[key] = tree

    def predict(self, features):
        if self.node_type == 'leaf' or (features[self.feature] not in self.dict):
            return self.Class

        tree = self.dict.get(features[self.feature])
        return tree.predict(features)


# 计算数据集x的经验熵H(x)
def calc_ent(x):
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent

# 计算条件熵H(y/x)
def calc_condition_ent(x, y):
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        temp_ent = calc_ent(sub_y)
        ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent

    return ent


# 计算信息增益
def calc_ent_grap(x, y):
    base_ent = calc_ent(y)
    condition_ent = calc_condition_ent(x, y)
    ent_grap = base_ent - condition_ent

    return ent_grap

#ID3算法
def recurse_train(train_set, train_label, features):
    LEAF = 'leaf'
    INTERNAL = 'internal'

    # 步骤1——如果训练集train_set中的所有实例都属于同一类Ck
    label_set = set(train_label)
    if len(label_set) == 1:
        return Tree(LEAF, Class=label_set.pop())

    # 步骤2——如果特征集features为空
    class_len = [(i, len(list(filter(lambda x: x == i, train_label)))) for i in range(class_num)]  # 计算每一个类出现的个数
    (max_class, max_len) = max(class_len, key=lambda x: x[1])

    if len(features) == 0:
        return Tree(LEAF, Class=max_class)

    # 步骤3——计算信息增益,并选择信息增益最大的特征
    max_feature = 0
    max_gda = 0
    D = train_label
    for feature in features:
        A = np.array(train_set[:, feature].flat)  # 选择训练集中的第feature列（即第feature个特征）
        gda = calc_ent_grap(A, D)
        if gda > max_gda:
            max_gda, max_feature = gda, feature

    # 步骤4——信息增益小于阈值
    if max_gda < epsilon:
        return Tree(LEAF, Class=max_class)

    # 步骤5——构建非空子集
    sub_features = list(filter(lambda x: x != max_feature, features))
    tree = Tree(INTERNAL, feature=max_feature)

    max_feature_col = np.array(train_set[:, max_feature].flat)
    feature_value_list = set(
        [max_feature_col[i] for i in range(max_feature_col.shape[0])])  # 保存信息增益最大的特征可能的取值 (shape[0]表示计算行数)
    for feature_value in feature_value_list:

        index = []
        for i in range(len(train_label)):
            if train_set[i][max_feature] == feature_value:
                index.append(i)

        sub_train_set = train_set[index]
        sub_train_label = train_label[index]

        sub_tree = recurse_train(sub_train_set, sub_train_label, sub_features)
        tree.add_tree(feature_value, sub_tree)

    return tree

#训练环节
def train(train_set, train_label, features):
    return recurse_train(train_set, train_label, features)

#预测环节
def predict(test_set, tree):
    result = []
    for features in test_set:
        tmp_predict = tree.predict(features)
        result.append(tmp_predict)
    return np.array(result)

'''C4.5算法
def recurse_train(train_set, train_label, features):
    LEAF = 'leaf'
    INTERNAL = 'internal'

    # 步骤1——如果训练集train_set中的所有实例都属于同一类Ck
    label_set = set(train_label)
    if len(label_set) == 1:
        return Tree(LEAF, Class=label_set.pop())

    # 步骤2——如果特征集features为空
    class_len = [(i, len(list(filter(lambda x: x == i, train_label)))) for i in range(class_num)]  # 计算每一个类出现的个数
    (max_class, max_len) = max(class_len, key=lambda x: x[1])

    if len(features) == 0:
        return Tree(LEAF, Class=max_class)

    # 步骤3——计算信息增益,并选择信息增益最大的特征
    max_feature = 0
    max_gda = 0
    D = train_label
    for feature in features:
        A = np.array(train_set[:, feature].flat)  # 选择训练集中的第feature列（即第feature个特征）
        gda = calc_ent_grap(A, D)
        if calc_ent(A) != 0:  # 计算信息增益比(与ID3的区别)
            gda /= calc_ent(A)
        if gda > max_gda:
            max_gda, max_feature = gda, feature

    # 步骤4——信息增益小于阈值
    if max_gda < epsilon:
        return Tree(LEAF, Class=max_class)

    # 步骤5——构建非空子集
    sub_features = list(filter(lambda x: x != max_feature, features))
    tree = Tree(INTERNAL, feature=max_feature)

    max_feature_col = np.array(train_set[:, max_feature].flat)
    feature_value_list = set(
        [max_feature_col[i] for i in range(max_feature_col.shape[0])])  # 保存信息增益最大的特征可能的取值 (shape[0]表示计算行数)
    for feature_value in feature_value_list:

        index = []
        for i in range(len(train_label)):
            if train_set[i][max_feature] == feature_value:
                index.append(i)

        sub_train_set = train_set[index]
        sub_train_label = train_label[index]

        sub_tree = recurse_train(sub_train_set, sub_train_label, sub_features)
        tree.add_tree(feature_value, sub_tree)

    return tree


def train(train_set, train_label, features):
    return recurse_train(train_set, train_label, features)


def predict(test_set, tree):
    result = []
    for features in test_set:
        tmp_predict = tree.predict(features)
        result.append(tmp_predict)
    return np.array(result)
'''

class_num = 10  # MNIST数据集有十个标签
feature_len = 784  #MNIST数据集每个图像有784个特征
epsilon = 0.001  # 设定阈值

if __name__ == '__main__':

    print("Start read data...")
    kf = KFold(n_splits=5, random_state=None)
    kfoldscore = []
    kfoldrecall = []
    kfoldpre = []
    time_1 = time.time()

    #读取数据集
    raw_data = pd.read_csv('../data/mnist_784.csv', header=0)
    data = raw_data.values
    character = data[0::, 1::]
    features = binaryzation_features(character)  # 图片二值化
    labels = data[::, 0]
    # 选取数据集数量
    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=60000, test_size=10000,
                                                        random_state=10)

        # 显示5折交叉验证的具体划分情况
    for train_index, test_index in kf.split(x_train, y_train):
        print("Train:", train_index, "Validation:", test_index)
        X_train, X_test = x_train[train_index], x_train[test_index]
        Y_train, Y_test = y_train[train_index], y_train[test_index]

    i = 1
    for train_index, test_index in kf.split(x_train, y_train):
        time_2 = time.time()
        print('\n{} of kfold {}'.format(i, kf.n_splits))
        X_train, X_test = x_train[train_index], x_train[test_index]
        Y_train, Y_test = y_train[train_index], y_train[test_index]
        tree = train(X_train, Y_train, list(range(feature_len)))
        time_3 = time.time()
        print('训练耗时 %f seconds' % (time_3 - time_2))

        print('开始预测')
        test_predict = predict(X_test, tree)
        time_4 = time.time()
        print('预测耗时 %f seconds' % (time_4 - time_3))

        for k in range(len(test_predict)):
            if test_predict[k] == None:
                test_predict[k] = epsilon
        score = accuracy_score(Y_test, test_predict)
        recall=recall_score(Y_test, test_predict,average='macro')
        precision=precision_score(Y_test, test_predict,average='macro')
        print("此折的准确率为 %f" % score)
        print("此折的召回率为 %f" % recall)
        print("此折的为精确度 %f" % precision)
        kfoldscore.append(score)
        kfoldrecall.append(recall)
        kfoldpre.append(precision)
        i += 1
        # 输出准确率，kfoldscore是储存每一折准确率的数组，average是这一轮的平均准确率
    average = statistics.mean(kfoldscore)
    print("平均准确度为 %f" % average)
    averagerecall = statistics.mean(kfoldrecall)
    averagepre = statistics.mean(kfoldpre)
    x=[1,2,3,4,5]
    plt.plot(x,  kfoldscore)
    plt.plot(x, kfoldrecall)
    plt.plot(x, kfoldpre)
    plt.legend(['accuracy_score', 'recall_score', 'precision_score'], loc='upper right')
    plt.show()
    '''此步骤用于测试，删掉此行和最后一行即可运行
    测试集计算
    tree = train(x_train, y_train, list(range(feature_len)))
    test_predict = predict(x_test, tree)
    time_4 = time.time()
    print('预测耗时 %f seconds' % (time_4 - time_1))

    for k in range(len(test_predict)):
        if test_predict[k] == None:
            test_predict[k] = epsilon
    score = accuracy_score(y_test, test_predict)
    print("测试集上准确率为 ", score)
    '''












