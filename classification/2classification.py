# coding:utf-8

import matplotlib.pyplot as plt
import pandas as pd
from numpy import *
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


train_file = pd.read_csv(r'data\train_data_10_percent_corrected_classify.csv')  # 读取已处理的训练集文件
print("训练集维度：", train_file.shape)
col_num = train_file.shape[1]
train_file.columns = [i+1 for i in range(col_num)]  # 命名训练集文件的每列名称

feature_num = 41
x = train_file[[i+1 for i in range(feature_num)]]
y = train_file[[col_num]]  # 训练集最后一列

unlabeled_test_file = pd.read_csv(r'data\test_data_10_percent_corrected_classify_unlabeled.csv')
col_num1 = unlabeled_test_file.shape[1]
unlabeled_test_file.columns = [k+1 for k in range(col_num1)]
print("无标签的测试集维度：", unlabeled_test_file.shape)

labeled_test_file = pd.read_csv(r'data\test_data_10_percent_corrected_classify_labeled.csv')
labeled_test_file.columns = [j+1 for j in range(labeled_test_file.shape[1])]
col_num2 = labeled_test_file.shape[1]
y_test = labeled_test_file[[col_num2]]
print("带标签的测试集维度：", labeled_test_file.shape)

x_train = x
x_test = unlabeled_test_file
y_train = y

vec = DictVectorizer(sparse=False)
x_train = vec.fit_transform(x_train.to_dict(orient='record'))
x_test = vec.transform(x_test.to_dict(orient='record'))
print(vec.feature_names_, "\n", x_train[:-1])

dtc_accuracy_score = []
nbc_accuracy_score = []
rfc_accuracy_score = []
lgr_accuracy_score = []

dtc_auc = []
nbc_auc = []
rfc_auc = []
lgr_auc = []

for i in range(3):
    print('第' + str(i + 1) + '次实验：')

    # 决策树
    dtc = DecisionTreeClassifier()
    dtc = dtc.fit(x_train, y_train)
    dtc_y_pre = dtc.predict(x_test)

    fpr1, tpr1, thresholds = metrics.roc_curve(y_test, dtc_y_pre)
    auc = metrics.auc(fpr1, tpr1)
    plt.subplot(2, 2, 1)
    plt.plot(fpr1, tpr1, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc)
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title('Decision Tree AUC', fontsize=10)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=10)
    dtc_accuracy_score.append(accuracy_score(y_test, dtc_y_pre))
    dtc_auc.append(auc)

    print("决策树：", "\n", classification_report(dtc_y_pre, y_test, target_names=["normal", "abnormal"]))
    print("决策树AUC：", roc_auc_score(y_test, dtc_y_pre))

    # 朴素贝叶斯
    nbc = GaussianNB()
    nbc = nbc.fit(x_train, y)
    nbc_y_pre = nbc.predict(x_test)

    fpr2, tpr2, thresholds1 = metrics.roc_curve(y_test, nbc_y_pre)
    auc = metrics.auc(fpr2, tpr2)
    plt.subplot(2, 2, 2)
    plt.plot(fpr2, tpr2, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc)
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title('Naive Bayes AUC', fontsize=10)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=10)
    nbc_accuracy_score.append(accuracy_score(y_test, nbc_y_pre))
    nbc_auc.append(auc)

    print("朴素贝叶斯：", "\n", classification_report(nbc_y_pre, y_test, target_names=["normal", "abnormal"]))
    print("朴素贝叶斯AUC：", roc_auc_score(y_test, nbc_y_pre))

    # 随机森林
    rfc = RandomForestClassifier()
    rfc = rfc.fit(x_train, np.array(y).ravel())
    rfc_y_pre = rfc.predict(x_test)
    fpr3, tpr3, thresholds = metrics.roc_curve(y_test, rfc_y_pre)
    auc = metrics.auc(fpr3, tpr3)
    plt.subplot(2, 2, 3)
    plt.plot(fpr3, tpr3, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc)
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title('Random Forest AUC', fontsize=10)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=10)
    rfc_accuracy_score.append(accuracy_score(y_test, rfc_y_pre))
    rfc_auc.append(auc)

    print("随机森林：", "\n", classification_report(rfc_y_pre, y_test, target_names=["normal", "abnormal"]))
    print("随机森林AUC：", roc_auc_score(y_test, rfc_y_pre))

    # 逻辑回归
    lgr = LogisticRegression()
    lgr = lgr.fit(x_train, y)
    lgr_y_pre = lgr.predict(x_test)
    fpr4, tpr4, thresholds = metrics.roc_curve(y_test, lgr_y_pre)
    auc = metrics.auc(fpr4, tpr4)
    plt.subplot(2, 2, 4)
    plt.plot(fpr4, tpr4, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc)
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title('Logistic Regression AUC', fontsize=10)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=10)
    lgr_accuracy_score.append(accuracy_score(y_test, lgr_y_pre))
    lgr_auc.append(auc)

    print("逻辑回归：", "\n", classification_report(lgr_y_pre, y_test, target_names=["normal", "abnormal"]))
    print("逻辑回归AUC：", roc_auc_score(y_test, lgr_y_pre))


print("决策树分类精确度：", mean(dtc_accuracy_score))
print("朴素贝叶斯分类精确度：", mean(nbc_accuracy_score))
print("随机森林分类精确度：", mean(rfc_accuracy_score))
print("逻辑回归分类精确度：", mean(lgr_accuracy_score))
print('--------')
print("决策树AUC：", mean(dtc_auc))
print("朴素贝叶斯AUC：", mean(nbc_auc))
print("随机森林AUC：", mean(rfc_auc))
print("逻辑回归AUC：", mean(lgr_auc))
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
plt.show()







