__author__ = 'jennytou'

from sklearn.neighbors import KNeighborsClassifier

def train(X_train, Y_train,k):
    neigh = KNeighborsClassifier(n_neighbors=int(k))
    neigh.fit(X_train, Y_train)
    return neigh

def test(X_test, neigh):
    return neigh.predict(X_test)