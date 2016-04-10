__author__ = 'jennytou'
import numpy as np
import matplotlib.pyplot as plt

#plot individual features out to have a sense of which is useful
def plot_1d(X_train, Y_train):
    ctr = []
    alz = []
    count = 0
    print Y_train
    for y in Y_train:
        if y == 0:
            ctr.append(X_train[count][1])
        else:
            alz.append(X_train[count][1])
        count = count + 1
    print ctr
    print alz
    plt.title('feature 1- BI')
    #plt.axis([0.02, 0.1, -1, 2])
    plt.plot(ctr,len(ctr)*[0], 'x',color = 'red')
    plt.plot(alz,len(alz)*[1], 'x',color = 'blue')
    plt.show()