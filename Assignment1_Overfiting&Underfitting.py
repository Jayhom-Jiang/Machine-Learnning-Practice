#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:
@file: A1_2021.py
@Software: PyCharm
@time: 2021/9/28
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

os.makedirs("Pics", exist_ok=True)
os.makedirs("Pics/Pics_M9/", exist_ok=True)

# calculate MAE
def cal_MAE(y, y_pre):
    error = abs((y-y_pre)).sum()/y.shape[0]
    return error

# generate data
def DataGenerate(train_num, valid_num, M):
    x_train = np.linspace(0., 1., train_num)  # training set
    x_valid = np.linspace(0., 1., valid_num)  # validation set
    np.random.seed(640)
    t_train = np.sin(4 * np.pi * x_train) + 0.3 * np.random.randn(train_num)
    t_valid = np.sin(4 * np.pi * x_valid) + 0.3 * np.random.randn(valid_num)
    if M == 9:
        x_train = x_train.reshape(train_num, 1)
        x_valid = x_valid.reshape(valid_num, 1)
        t_train = t_train.reshape(train_num, 1)
        t_valid = t_valid.reshape(valid_num, 1)
    return x_train, t_train, x_valid, t_valid

# plot figure (fitting situation under different M)
def plot_fig(x, y, prediction, Training_or_Validation, M, lamb):
    plt.figure(figsize=(20, 10))
    plt.scatter(x, y, color='b', label=Training_or_Validation + ' Examples')
    plt.scatter(x, prediction, color='m', label='Prediction Examples')
    plt.plot(x, y, color='g', label='f-true(x)')
    plt.plot(x, prediction, 'r-', label='f-M(x)')
    plt.title(Training_or_Validation + ': M=' + str(M), fontsize=20)
    plt.xlabel('x', fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylabel('t', fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig('./Pics/' + Training_or_Validation + str(M) + '_' + str(lamb) + '.png')
    plt.close()

def train(x_train, t_train, M, lamb):
    if M == 9:
        poly_features_d = PolynomialFeatures(degree=M, include_bias=False)
        x_poly_d = poly_features_d.fit_transform(x_train)
        model = Ridge(alpha=lamb, solver="cholesky")
        model.fit(x_poly_d, t_train)
        x_plot_poly = poly_features_d.fit_transform(x_train)
        prediction = np.dot(x_plot_poly, model.coef_.T) + model.intercept_
        f = 'Ridge'
    else:
        model = np.polyfit(x_train, t_train, M)
        prediction = np.polyval(model, x_train)
        f = np.poly1d(model)
    mean_error = cal_MAE(t_train, prediction)
    return model, mean_error, prediction, f

def valid(x_valid, t_valid, model, M):
    if M == 9:
        poly_features_d = PolynomialFeatures(degree=M, include_bias=False)
        x_plot_poly = poly_features_d.fit_transform(x_valid)
        prediction = np.dot(x_plot_poly, model.coef_.T) + model.intercept_
        R_square = r2_score(t_valid, prediction)
    else:
        prediction = np.polyval(model, x_valid)
        f = np.poly1d(model)
        R_square = r2_score(t_valid, f(x_valid))

    mean_error = cal_MAE(t_valid, prediction)
    return mean_error, R_square, prediction

# plot figure (MAE under different M)
def plot_error_fig(train_error_list, valid_error_list):
    plt.figure(figsize=(12, 6))
    x_plot = list(range(0, 10))
    plt.scatter(x_plot, train_error_list[0:10], color='b', label='Training Error')
    plt.scatter(x_plot, valid_error_list[0:10], color='m', label='Validation Error')
    plt.plot(x_plot, train_error_list[0:10], color='g', label='Training')
    plt.plot(x_plot, valid_error_list[0:10], 'r-', label='Validation')
    plt.scatter(9, train_error_list[10], color='b', marker='x', label='Training Error(lamda=1)')
    plt.scatter(9, train_error_list[11], color='b', marker='v', label='Training Error(lamda=1)')
    plt.scatter(9, valid_error_list[10], color='m', marker='x', label='Validation Error(lamda=5e-14)')
    plt.scatter(9, valid_error_list[11], color='m', marker='v', label='Validation Error(lamda=5e-14)')
    plt.title('MAE of Training and Validation under different M')
    plt.xlabel('M')
    plt.xticks()
    plt.ylabel('MAE')
    plt.yticks()
    plt.legend()
    plt.savefig('./Pics/MAE_M.png')
    plt.close()

# plot figure (learning curve under different lambda)
def plot_curve_lamb(M):
    if M != 9:
        return
    else:
        arr = (np.array(list(range(-15000, 1)))) / 1000
        lamb_plot = []
        train_error_lamb = []
        valid_error_lamb = []
        R_square_lamb = []
        for i in range(len(arr)):
            lamb_plot.append(10 ** arr[i])
        for i in range(len(arr)):
            lamb = lamb_plot[i]
            x_train, t_train, x_valid, t_valid = DataGenerate(train_num, valid_num, M)
            model, train_error, train_prediction, f_train = train(x_train, t_train, M, lamb)
            valid_error, R_square, valid_prediction = valid(x_valid, t_valid, model, M)
            train_error_lamb.append(train_error)
            valid_error_lamb.append(valid_error)
            R_square_lamb.append(R_square)

        min_index = valid_error_lamb.index(min(valid_error_lamb))
        print('The best model | Lambda = ', lamb_plot[min_index], '| Training error = ', train_error_lamb[min_index],
              '| Validation error = ', valid_error_lamb[min_index], '| R-square: ', R_square_lamb[min_index])

        plt.figure(figsize=(12, 6))
        plt.scatter(lamb_plot[min_index], train_error_lamb[min_index], color='b', label='Best Model: Training Error')
        plt.scatter(lamb_plot[min_index], valid_error_lamb[min_index], color='m', label='Best Model: Validation Error')
        plt.plot(lamb_plot, train_error_lamb, color='g', label='Training')

        plt.plot(lamb_plot, valid_error_lamb, 'r-', label='Validation')
        plt.semilogx()
        plt.title('MAE of Training and Validation under different lambda')
        plt.xlabel('Lambda')
        plt.xticks()
        plt.ylabel('MAE')
        plt.yticks()
        plt.legend()
        plt.savefig('./Pics/Pics_M9/MAE_lambda.png')
        plt.close()

lamb_list = [1.67e-9, 1e-3, 5e-14]  # well-fitting, underfitting, overfitting
train_num = 10     # numbers of train samples
valid_num = 100    # numbers of valid samples
train_error_list = []
valid_error_list = []
R_square_list = []
for i in range(12):
    if i < 9:
        M = i
        lamb = None
        print('M = ', M)
    else:
        M = 9
        lamb = lamb_list[i-9]
    x_train, t_train, x_valid, t_valid = DataGenerate(train_num, valid_num, M)
    model, train_error, train_prediction, f_train = train(x_train, t_train, M, lamb)
    plot_fig(x_train, t_train, train_prediction, 'Training', M, lamb)
    valid_error, R_square, valid_prediction = valid(x_valid, t_valid, model, M)
    plot_fig(x_valid, t_valid, valid_prediction, 'Validation', M, lamb)

    print('f' + str(M) + '(x) = ')
    print(f_train)
    print('Training Error:', train_error, '| Validation Error:', valid_error)
    print('R_square: {:.2f}'.format(R_square))
    print('----------------------------------------------------------------------------------------------------')
    train_error_list.append(train_error)
    valid_error_list.append(valid_error)
    R_square_list.append(R_square)

plot_error_fig(train_error_list, valid_error_list)
plot_curve_lamb(9)




