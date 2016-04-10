__author__ = 'jennytou'
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#plot individual features out to have a sense of which is useful
def plot_1d(X_train, Y_train):
    ctr = []
    alz = []
    count = 0
    print Y_train
    for y in Y_train:
        if y == 0:
            ctr.append(X_train[count][12])
        else:
            alz.append(X_train[count][12])
        count = count + 1
    #print ctr
    #print alz
    plt.title('feature 12- min_diff_f1f2')
    plt.axis([0, 1, -1, 2])
    plt.plot(ctr,len(ctr)*[0], 'x',color = 'red')
    plt.plot(alz,len(alz)*[1], 'x',color = 'blue')
    plt.show()

def plot_3d(X_train, y_train):
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()

    #separate control and alzheimer's
    ctr = []
    alz = []
    count = 0
    for y in y_train:
        if y == 0:
            ctr.append(X_train[count])
        else:
            alz.append(X_train[count])
        count = count + 1

    ctr = np.array(ctr)
    alz = np.array(alz)

    plt.title('Data Distribution after PCA')

    ax.scatter(ctr[:, 0], ctr[:, 1], ctr[:, 2], c='b', cmap=plt.cm.spectral)
    ax.scatter(alz[:, 0], alz[:, 1], alz[:, 2], c='r', cmap=plt.cm.spectral)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()