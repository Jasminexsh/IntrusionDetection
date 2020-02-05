# encoding:utf-8

from numpy import *
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import GradientBoostingClassifier


train_file = pd.read_csv(r'data\train_data_10_percent_corrected_multi_classify.csv')  # 读取已处理的训练集文件
col_num = train_file.shape[1]  # 训练集文件的列数目
train_file.columns = [i+1 for i in range(col_num)]  # 命名训练集文件的每列名称
print("训练集维度：", train_file.shape)

# x = kdd99[[i+1 for i in range(col_num-1)]]  # 切分训练集的41列特征
feature_num = 41  # 用于训练模型的特征数目
x = train_file[[i+1 for i in range(feature_num)]]
y = train_file[[col_num]]  # 训练集最后一列：异常类型

unlabeled_test_file = pd.read_csv(r'data\test_data_10_percent_corrected_multi_classify_unlabeled.csv')
col_num1 = unlabeled_test_file.shape[1]
unlabeled_test_file.columns = [k+1 for k in range(col_num1)]
print("无标签测试集的维度：", unlabeled_test_file.shape)

labeled_test_file = pd.read_csv(r'data\test_data_10_percent_corrected_multi_classify_labeled.csv')
labeled_test_file.columns = [j+1 for j in range(labeled_test_file.shape[1])]
col_num2 = labeled_test_file.shape[1]
y_test = labeled_test_file[[col_num2]]
print("带标签测试集的维度：", labeled_test_file.shape)

x_train = x
x_test = unlabeled_test_file
y_train = y

vec = DictVectorizer(sparse=False)
x_train = vec.fit_transform(x_train.to_dict(orient='record'))
x_test = vec.transform(x_test.to_dict(orient='record'))
print(vec.feature_names_, "\n", x_train[:-1])

"""dtc = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None, max_features=None,
                             max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                             min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False,
                             random_state=None, splitter='best')"""

# 建立空列表，将每次的精确度存入列表中，便于求解多次的平均值。
dtc_accuracy_score = []
nbc_accuracy_score = []
rfc_accuracy_score = []
lgr_accuracy_score = []

# 迭代三次，检验分类器模型的优劣。
for i in range(3):
    print('第'+str(i+1)+'次实验：')
    # 决策树分类器
    dtc = DecisionTreeClassifier()
    dtc = dtc.fit(x_train, y_train)
    dtc_y_pre = dtc.predict(x_test)
    dtc_accuracy_score.append(accuracy_score(dtc_y_pre, y_test))
    # print("决策树分类精确度：", accuracy_score(dtc_y_pre, y_test))
    print("决策树：", "\n", classification_report(dtc_y_pre, y_test, target_names=["normal", "smurf", "others"]))
    # print(dtc_accuracy_score)

    # 朴素贝叶斯分类器
    nbc = GaussianNB()
    nbc = nbc.fit(x_train, y_train)
    nbc_y_pre = nbc.predict(x_test)
    nbc_accuracy_score.append(accuracy_score(nbc_y_pre, y_test))
    # print("朴素贝叶斯分类精确度：", accuracy_score(y_test, nbc_y_pre))
    print("朴素贝叶斯：",  "\n", classification_report(nbc_y_pre, y_test, target_names=["normal", "smurf", "others"]))

    # 随机森林
    rfc = RandomForestClassifier(n_estimators=150, random_state=2)
    rfc = rfc.fit(x_train, np.array(y_train).ravel())
    rfc_y_pre = rfc.predict(x_test)
    rfc_accuracy_score.append(accuracy_score(rfc_y_pre, y_test))
    # print("随机森林分类精确度：", accuracy_score(y_test, rfc_y_pre))
    print("随机森林：", "\n", classification_report(rfc_y_pre, y_test, target_names=["normal", "smurf", "others"]))

    # 逻辑回归
    lgr = LogisticRegression()
    lgr = lgr.fit(x_train, np.array(y_train).ravel())
    lgr_y_pre = lgr.predict(x_test)
    lgr_accuracy_score.append(accuracy_score(lgr_y_pre, y_test))
    # print("逻辑回归分类精确度：", accuracy_score(y_test, lgr_y_pre))
    print("逻辑回归：", "\n", classification_report(lgr_y_pre, y_test, target_names=["normal", "smurf", "others"]))

# 3次迭代后，不同分类器模型的平均精确度指标值。
print("决策树分类精确度：", mean(dtc_accuracy_score))
print("朴素贝叶斯分类精确度：", mean(nbc_accuracy_score))
print("随机森林分类精确度：", mean(rfc_accuracy_score))
print("逻辑回归分类精确度：", mean(nbc_accuracy_score))

# 运行速度较慢 不推荐
"""# 梯度提升决策树
    gbc = GradientBoostingClassifier(random_state=10, subsample=0.8)
    gbc = gbc.fit(x_train, np.array(y_train).ravel())
    gbc_y_pre = gbc.predict(x_test)
    print("梯度提升决策树精确度：", accuracy_score(y_test, gbc_y_pre))
    print("梯度提升决策树：", "\n", classification_report(gbc_y_pre, y_test, target_names=["normal", "smurf", "others"]))"""



