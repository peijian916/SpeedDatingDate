# -*- coding: utf-8 -*-
'''
    create on 2019.03.31
    @author: Jian Pei
'''

import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split

from pandas_tools import inspect_dataset, process_missing_data, get_pair_data
from ml_tools import select_features, balance_samples, train_model, print_result, plot_roc

#是否处理非平衡数据
is_process_unbalanced_data = True

#是否交叉验证
is_cv = True

#是否进行特征选择
is_feat_select = True

#设置随机种子
random_seed = 7
np.random.seed(random_seed)


file_path = './dataset/SpeedDatingData.csv'

def run_main():
    df_data = pd.read_csv(file_path, encoding='gbk')
    # inspect_dataset(df_data)

    process_missing_data(df_data)

    pair_data, labels, features = get_pair_data(df_data)
    # print(pair_data)
    # print(label)
    # print(features)

    #进行特征选择
    if is_feat_select:
        pair_data, selected_features = select_features(pair_data, labels, features)
        print('选择的特征：')
        print(selected_features)

    n_pos_samples = labels[labels == 1].shape[0]
    n_neg_samples = labels[labels == 0].shape[0]

    print('正样本数：{}'.format(n_pos_samples))
    print('负样本数：{}'.format(n_neg_samples))

    #处理非平衡数据
    if is_process_unbalanced_data:
        pair_data, labels = balance_samples(pair_data, labels)

    print(pair_data.shape)
    print(labels.shape)

    #分割训练集和测试集
    X_train, X_test, y_train ,y_test = train_test_split(pair_data, labels, test_size=0.1, random_state=random_seed)


    #训练模型，测试模型
    print('逻辑回归模型：')
    logist_model = train_model(X_train, y_train, model_name='logistic_regression', is_cv=True)
    logist_model_predictions =logist_model.predict(X_test)
    logist_model_prob_predictions = logist_model.predict_proba(X_test)
    print('逻辑回归模型预测结果：')
    print_result(y_test, logist_model_predictions, logist_model_prob_predictions)

    #绘制逻辑回归模型ROC曲线
    plot_roc(y_test, logist_model_prob_predictions,
             fig_title='Logistic Regression',
             save_path='./image/lr_roc.png')

    print('SVM模型：')
    svm_model = train_model(X_train, y_train, model_name='SVM', is_cv=True)
    svm_model_predictions = svm_model.predict(X_test)
    svm_model_prob_predictions = svm_model.predict_proba(X_test)
    print('SVM模型预测结果：')
    print_result(y_test, svm_model_predictions, svm_model_prob_predictions)

    #绘制SVM模型ROC曲线
    plot_roc(y_test, svm_model_prob_predictions,
             fig_title='SVM',
             save_path='./image/svm_roc.png')

    print('RF模型：')
    rf_model = train_model(X_train, y_train, model_name='random_forest', is_cv=True)
    # print(rf_model.feature_importances_)
    rf_model_predictions = rf_model.predict(X_test)
    rf_model_prob_predictions = rf_model.predict_proba(X_test)
    print('RF模型预测结果：')
    print_result(y_test, rf_model_predictions, rf_model_prob_predictions)

    #绘制RF模型ROC曲线
    plot_roc(y_test, rf_model_prob_predictions,
             fig_title='RF',
             save_path='./image/rf_roc.png')



if __name__ == '__main__':
    run_main()

