import data
import SVM
import KNN
import visualize_features
from sklearn.mixture import GMM
import numpy as np
from sklearn import metrics

def cross_validate(X_scaled_reduced, Y, subjectID):
	# split training and testing data
	y_pred = []
	y_true = []
	for subject in subjectID:
		testList = []
		testList.append(subject)
		X_train, Y_train, X_test, Y_test, trainID, testID = data.split_train_test(X_scaled_reduced,Y,subjectID,testID=testList)

		#SVM
		#clf = SVM.train(X_train,Y_train)
		#y_pred.extend(SVM.test(X_test,clf))

		#KNN
		neigh = KNN.train(X_train,Y_train)
		y_pred.extend(KNN.test(X_test,neigh))
		y_true.extend(Y_test)
	precision = metrics.precision_score(y_true, y_pred)
	recall = metrics.recall_score(y_true,y_pred)
	f1 = metrics.f1_score(y_true,y_pred)

	return precision, recall, f1


if __name__ == "__main__":
	X, Y, subjectID = data.load_data("control_features_combinedSubject.txt", "dementia_features_combinedSubject.txt")
	X = data.get_useful_features_mat(X)

	alz_count = 0
	for y in Y:
		if y:
			alz_count = alz_count + 1
	print float(alz_count)/float(len(Y))

	#normalize features
	#print X
	X_scaled = data.normalize_features(X)
	#print X_scaled

	#PCA
	pca, explained_variance_ratio_, X_scaled_reduced = data.reduce_dimension(X_scaled)

    #plot out data 3D
	#visualize_features.plot_3d(X_scaled_reduced, Y)

	#cross validate
	precision, recall, f1 = cross_validate(X_scaled_reduced, Y, subjectID)

	print precision, recall, f1

	#visualize features
	#visualize_features.plot_1d(X_train,Y_train)