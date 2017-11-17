import math
import numpy as np
import random as rnd
from matplotlib import colors
import matplotlib.pyplot as plt

def find_weights(p1, p2):
    slope = (p2[1]-p1[1])/(p2[0]-p1[0])
    intercept = p1[1] - slope*p1[0]
    return intercept, slope

def epoch(X, y, stop_lim, learn_rate):
    epoch_change = 1
    epoch_iter = 1
    w_start = np.array([0, 0, 0])
    while (epoch_change >= learn_rate):
        idx = rnd.sample(range(0, 500), 500)
        w_next = w_start
        for i in idx:
            grad = sgrad(w_next, X[i], y[i])
            w_next = w_next - learning_rate * grad
        epoch_change = np.linalg.norm(w_next) - np.linalg.norm(w_start)
        # print(w_start, w_next)
        w_start = w_next
        # print("Iter {}: {}".format(epoch_iter,epoch_change))
        epoch_iter += 1
    return -w_next, epoch_iter-1

def signal(w, x):
    return np.dot(w, np.append([1], x))


def logistic_func(s):
    return math.exp(s)/(1+math.exp(s))

def theta(w,x):
    return logistic_func(signal(w,x))

def cost(w, x, y) -> float :
    return math.log(1 + math.exp(-y*signal(w,x)))

def sgrad(w,x,y):
    denominator = (1 + math.exp(-y * signal(w,x)))
    numerator = np.multiply(y, np.append([1],x))
    return np.divide(numerator,denominator)

learning_rate = 0.01

n_test = 1000
rep = 1
total = n_test * rep
steps_total = 0
correct_total = 0
cross_entr_total = 0.0
w_train = None

for i in range(0, rep):
    X_train = np.multiply(np.add(np.random.rand(500, 2), -0.5), 2)
    X_test = np.multiply(np.add(np.random.rand(n_test, 2), -0.5), 2)

    p1 = rnd.choice(X_train)
    p2 = rnd.choice(X_train)
    while (p1 == p2).all():
        p2 = rnd.choice(X_train)
    b, a = find_weights(p1, p2)

    y_train = []
    for i in range(0, 500):
        y_train.append(math.copysign(1, X_train[i][1] - (X_train[i][0] * a + b)))
    y_train = np.array(y_train)

    y_test = []
    cross_entr_sum = 0.0
    for i in range(0, n_test):
        y_test.append(math.copysign(1, X_test[i][1] - (X_test[i][0] * a + b)))
    y_test = np.array(y_test)
    w_train, iters = epoch(X_train,y_train, 0.007, learning_rate)

    y_predict = []
    for i in range(0, n_test):
        cross_entr_sum += cost(w_train, X_test[i], y_test[i])
        y_predict.append(theta(w_train, X_test[i])-0.66)
    y_predict = np.sign(np.array(y_predict))
    cross_entr = cross_entr_sum / n_test
    cross_entr_total += cross_entr
    correct = (y_predict == y_test).sum()
    print('steps: {}, error rate(test-set): {:f}, cost: {:f}'.format(iters, 1-correct/n_test, cross_entr))
    steps_total += iters
    correct_total += correct

print('AVG steps: {}, AVG error rate(test-set): {:f}, AVG cost: {:f}'.format(steps_total/(rep + 0.0), 1-correct_total/total, cross_entr_total/(rep + 0.0)))

X_train = X_test
y_train = y_test
for i in range(0,1000):
    if theta(w_train, X_train[i]) >= .66:
        plt.scatter(X_train[i:, 0], X_train[i:, 1], c='red')
    else:
        plt.scatter(X_train[i:, 0], X_train[i:, 1], c='blue')
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = b + a * x_vals
if (np.abs(y_vals) > [1.1,1.1]).any():
    y_vals = np.multiply(np.minimum(np.abs(y_vals), 1.1), np.sign(y_vals))
    x_vals =  (y_vals - b) / a
plt.plot(x_vals,y_vals,'r--')

plt.show()