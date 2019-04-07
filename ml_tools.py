# -*- coding: utf-8 -*-
'''
    create on 2019.03.31
    @author: Jian Pei
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import VarianceThreshold, SelectPercentile
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV

'''进行特征选择'''
def select_features(pair_data, labels, features):
    print('特征选择')

    #step.1 过滤掉"低方差"的特征列
    vt_sel = VarianceThreshold(threshold=(0.85 * (1 - 0.85)))
    vt_sel.fit(pair_data)

    #step. 需要过滤的特征
    sel_features_1 = features[vt_sel.get_support()] #get_support()获取特征筛选结果
    sel_pair_data_1 = pair_data[:, vt_sel.get_support()]
    print('"低方差"过滤掉{}个特征'.format(features.shape[0] - sel_features_1.shape[0]))

    #step.2 根据"单变量统计分析"选择特征
    #保留重要的95%特征
    sp_sel = SelectPercentile(percentile=95)
    sp_sel.fit(sel_pair_data_1, labels)

    sel_features_2 = sel_features_1[sp_sel.get_support()]
    sel_pair_data_2 = sel_pair_data_1[:, sp_sel.get_support()]
    print('"单变量统计分析"过滤掉{}个特征'.format(sel_features_1.shape[0] - sel_features_2.shape[0]))

    #根据特征的score绘制柱状图
    feat_ser = pd.Series(data=sp_sel.scores_, index=features)
    sorted_feat_ser = feat_ser.sort_index(ascending=False)
    plt.figure(figsize=(18, 12))
    sorted_feat_ser.plot(kind='bar')
    plt.savefig('./feat_importance.png')
    plt.show()

    return sel_pair_data_2, sel_features_2


'''处理非平衡数据集'''
def balance_samples(pair_data, labels):
    labels = labels.reshape((labels.size, 1))

    all_data = np.concatenate((pair_data, labels), axis=1)
    pos_data = all_data[all_data[:, -1] == 1]
    neg_data = all_data[all_data[:, -1] == 0]

    n_pos_samples = pos_data.shape[0]

    #已知负样本过多
    n_selected_neg_samples = int(n_pos_samples * 2)
    sampled_neg_data = neg_data[np.random.choice(neg_data.shape[0],n_selected_neg_samples)]
    sampled_all_data = np.concatenate((sampled_neg_data, pos_data))

    selected_pair_data = sampled_all_data[:, : -1]
    selected_labels = sampled_all_data[:, -1]

    return selected_pair_data, selected_labels



'''训练分类模型，默认为逻辑回归，默认不执行交叉验证'''
def train_model(X_train, y_train, model_name='logistic_regression', is_cv=False):
    #逻辑回归
    if model_name == 'logistic_regression':
        lr_model = linear_model.LogisticRegression()
        if is_cv:
            print('交叉验证：')
            params = {'C': [1e-4, 1e-3, 1e-2, 0.1, 1]}
            gs_model = GridSearchCV(lr_model, param_grid=params, cv=5,
                                    scoring='roc_auc', verbose=3)
            gs_model = gs_model.fit(X_train, y_train)
            print('最优参数：', gs_model.best_params_)
            best_model = gs_model.best_estimator_

        else:
            print('使用模型的参数：')
            lr_model.fit(X_train, y_train)
            best_model = lr_model
    #SVM
    elif model_name == 'SVM':
        svm_model = svm.SVC(probability=True)
        if is_cv:
            print('交叉验证：')
            params = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1e-5, 1e-4, 1e-3, 1e-2, 0.1]}

            gs_model = GridSearchCV(svm_model, param_grid=params, cv=5,
                                    scoring='roc_auc', verbose=3)
            gs_model.fit(X_train, y_train)
            print('最优参数：', gs_model.best_params_)
            best_model = gs_model.best_estimator_

        else:
            print('使用默认的参数：')
            svm_model.fit(X_train, y_train)
            best_model = svm_model

    #随机森林
    elif model_name == 'random_forest':
        rf_model = RandomForestClassifier()
        if is_cv:
            print('交叉验证：')
            params = {'n_estimators': [20, 40, 80, 100]}
            gs_model = GridSearchCV(rf_model, param_grid=params, cv=5,
                                    scoring='roc_auc', verbose=3)
            gs_model.fit(X_train, y_train)
            print('最优参数：',gs_model.best_params_)
            best_model = gs_model.best_estimator_

        else:
            print('使用默认的参数：')
            rf_model.fit(X_train, y_train)
            best_model = rf_model

    return best_model


'''输出预测准确率、AUC值'''
def print_result(true_labels, pred_labels, pred_probs):
    print('预测准确率：%.4f'%accuracy_score(true_labels, pred_labels))
    print('预测AUC值：%.4f'%roc_auc_score(true_labels, pred_probs[:, 1]))


'''绘制ROC曲线'''
def plot_roc(true_labels, pred_probs, fig_title='', save_path=''):
    false_positive_rate, true_positive_rate, _ = roc_curve(true_labels, pred_probs[:, 1], pos_label=1)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.figure(figsize=(16, 9))
    plt.title(fig_title)
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC=%4f'%roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.legend(loc='lower right')
    plt.xlabel('True Positive Rate')
    plt.ylabel('False Positive Rate')

    if save_path != '':
        plt.savefig(save_path)

    plt.show()
