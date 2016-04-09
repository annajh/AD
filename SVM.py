__author__ = 'jennytou'
print(__doc__)
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

def train(X_train, Y_train):
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train,Y_train)
    return clf

def test(X_test, clf):
    return clf.predict(X_test)

def plot(clf,X,Y):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    b = clf.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = clf.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])

    # plot the line, the points, and the nearest vectors to the plane
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(clf.support_vectors_, clf.support_vectors_,facecolors='red')
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

    plt.axis('tight')
    plt.show()