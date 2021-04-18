# __main__.py

from configparser import ConfigParser
from importlib import resources  # Python 3.7+
import sys

from reader import feed
from reader import viewer

def main():
    # od - Original data,sd - shuffle data, iv - independent varaible, dwiv - data without iv, ldwiv list of dwiv, cd - combined data.
    # "C:\\Users\\prav\\PycharmProjects\\SVM\\SVM\\Iris.csv"

    import pandas as pd
    import random
    import numpy as np
    def linearreg(a, b):
        od = pd.read_csv(a)
        od = od.drop(['Id'], axis=1)

        od = od.drop(od.index[range(100, 150)])  ## Drop the rows with iv values Iris-virginica

        sd = od.sample(frac=1)

        Y = []
        iv = sd['Species']

        for val in iv:
            if (val == 'Iris-setosa'):
                Y.append(0)
            else:
                Y.append(1)
        dwiv = sd.drop(['Species'], axis=1)
        ldwiv = dwiv.values.tolist()
        cd = pd.DataFrame(ldwiv)
        cd['species'] = Y  # adding back species for shuffle

        X = ldwiv

        x_train = []
        y_train = []
        x_test = []
        y_test = []

        sep = int(b * (len(X)))
        rem = int(len(X) - sep)
        x_train = X[0:sep]
        x_test = X[sep:]
        y_train = Y[0:sep]
        y_test = Y[sep:]
        print(len(x_train), len(x_test))
        print(len(y_train), len(y_test))
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        x_1 = x_train[:, 0]
        x_2 = x_train[:, 1]
        x_3 = x_train[:, 2]
        x_4 = x_train[:, 3]

        x_1 = np.array(x_1)
        x_2 = np.array(x_2)
        x_3 = np.array(x_3)
        x_4 = np.array(x_4)

        x_1 = x_1.reshape(sep, 1)
        x_2 = x_2.reshape(sep, 1)
        x_3 = x_3.reshape(sep, 1)
        x_4 = x_4.reshape(sep, 1)

        y_train = y_train.reshape(sep, 1)

        def sigmoid(x):
            return (1 / (1 + np.exp(-x)))

        m = sep
        alpha = 0.0001

        theta_0 = np.zeros((m, 1))
        theta_1 = np.zeros((m, 1))

        theta_2 = np.zeros((m, 1))
        theta_3 = np.zeros((m, 1))
        theta_4 = np.zeros((m, 1))

        epochs = 0
        cost_func = []
        while (epochs < 10000):
            y = theta_0 + theta_1 * x_1 + theta_2 * x_2 + theta_3 * x_3 + theta_4 * x_4
            y = sigmoid(y)

            cost = (- np.dot(np.transpose(y_train), np.log(y)) - np.dot(np.transpose(1 - y_train), np.log(1 - y))) / m

            theta_0_grad = np.dot(np.ones((1, m)), y - y_train) / m
            theta_1_grad = np.dot(np.transpose(x_1), y - y_train) / m
            theta_2_grad = np.dot(np.transpose(x_2), y - y_train) / m
            theta_3_grad = np.dot(np.transpose(x_3), y - y_train) / m
            theta_4_grad = np.dot(np.transpose(x_4), y - y_train) / m

            theta_0 = theta_0 - alpha * theta_0_grad
            theta_1 = theta_1 - alpha * theta_1_grad
            theta_2 = theta_2 - alpha * theta_2_grad
            theta_3 = theta_3 - alpha * theta_3_grad
            theta_4 = theta_4 - alpha * theta_4_grad

            cost_func.append(cost)
            epochs += 1

        from sklearn.metrics import accuracy_score

        test_x_1 = x_test[:, 0]
        test_x_2 = x_test[:, 1]
        test_x_3 = x_test[:, 2]
        test_x_4 = x_test[:, 3]

        test_x_1 = np.array(test_x_1)
        test_x_2 = np.array(test_x_2)
        test_x_3 = np.array(test_x_3)
        test_x_4 = np.array(test_x_4)

        test_x_1 = test_x_1.reshape(rem, 1)
        test_x_2 = test_x_2.reshape(rem, 1)
        test_x_3 = test_x_3.reshape(rem, 1)
        test_x_4 = test_x_4.reshape(rem, 1)

        index = list(range(rem, sep))

        theta_0 = np.delete(theta_0, index)
        theta_1 = np.delete(theta_1, index)
        theta_2 = np.delete(theta_2, index)
        theta_3 = np.delete(theta_3, index)
        theta_4 = np.delete(theta_4, index)

        theta_0 = theta_0.reshape(rem, 1)
        theta_1 = theta_1.reshape(rem, 1)
        theta_2 = theta_2.reshape(rem, 1)
        theta_3 = theta_3.reshape(rem, 1)
        theta_4 = theta_4.reshape(rem, 1)

        y_pred = theta_0 + theta_1 * test_x_1 + theta_2 * test_x_2 + theta_3 * test_x_3 + theta_4 * test_x_4
        y_pred = sigmoid(y_pred)

        new_y_pred = []
        for val in y_pred:
            if (val >= 0.5):
                new_y_pred.append(1)
            else:
                new_y_pred.append(0)

        print(accuracy_score(y_test, new_y_pred))

    linearreg("C:\\Users\\prav\\PycharmProjects\\SVM\\SVM\\Iris.csv", 0.5)

if __name__ == "__main__":
    main()