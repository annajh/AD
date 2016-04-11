__author__ = 'jennytou'

from sklearn.neighbors import KNeighborsClassifier

def train(X_train, Y_train):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, Y_train)
    return neigh

def test(X_test, neigh):
    return neigh.predict(X_test)