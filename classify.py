import data
import SVM
import KNN
import visualize_features
import UI
from sklearn.mixture import GMM
import numpy as np
from sklearn import metrics
import Tkinter as tk


def cross_validate(method, k, kernal,n, message):
	y_pred = []
	y_true = []
	X, Y, subjectID = data.load_data("control_features_per_visit.txt", "dementia_features_per_visit.txt")
	#No PCA, SVM, linear
	print len(subjectID)
	for subject in subjectID:
		try:
			result, truth = UI.classify_function(subject, method, k, kernal,n)
			y_pred.append(result)
			y_true.append(truth)
		except data.ValidationError as e:
			print e.message
			continue
	#print y_pred, y_true
	precision = metrics.precision_score(y_true, y_pred)
	recall = metrics.recall_score(y_true,y_pred)
	f1 = metrics.f1_score(y_true,y_pred)
	print message
	print precision, recall, f1


if __name__ == "__main__":
	'''
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

	#print precision, recall, f1
	'''

	root = tk.Tk()
	UI.Example(root).pack(fill="both", expand=True)
	root.mainloop()

'''
	#n-fold test
	#No PCA, SVM, linear
	cross_validate(2, 0, 1, -1, "No PCA, SVM, linear")

	#No PCA, SVM, rbf
	cross_validate(2, 0, 2, -1, "No PCA, SVM, rbf")

	#PCA = 3, SVM, linear
	cross_validate(2, 0, 1, 3, "PCA = 3, SVM, linear")

	#PCA = 3, SVM, rbf
	cross_validate(2, 0, 2, 3, "PCA = 3, SVM, rbf")

	#PCA = 5, SVM, linear
	cross_validate(2, 0, 1, 5, "PCA = 5, SVM, linear")

	#PCA = 5, SVM, rbf
	cross_validate(2, 0, 2, 5, "PCA = 5, SVM, rbf")

	#PCA = 7, SVM, linear
	cross_validate(2, 0, 1, 7, "PCA = 7, SVM, linear")

	#PCA = 7, SVM, rbf
	cross_validate(2, 0, 2, 7, "PCA = 7, SVM, rbf")

	#No PCA, KNN, k = 1
	cross_validate(1, 1, 1, -1, "No PCA, KNN, k = 1")

	#No PCA, KNN, k = 3
	cross_validate(1, 3, 1, -1, "No PCA, KNN, k = 3")

	#No PCA, KNN, k = 5
	cross_validate(1, 5, 1, -1, "No PCA, KNN, k = 5")

	#No PCA, KNN, k = 7
	cross_validate(1, 7, 1, -1, "No PCA, KNN, k = 7")

	#No PCA, KNN, k = 10
	cross_validate(1, 10, 1, -1, "No PCA, KNN, k = 10")

	#PCA 3, KNN, k = 1
	cross_validate(1, 1, 1, 3, "PCA = 3, KNN, k = 1")

	#PCA = 3, KNN, k = 3
	cross_validate(1, 3, 1, 3, "PCA = 3, KNN, k = 3")

	#PCA = 3, KNN, k = 5
	cross_validate(1, 5, 1, 3, "PCA = 3, KNN, k = 5")

	#PCA = 3, KNN, k = 7
	cross_validate(1, 7, 1, 3, "PCA = 3, KNN, k = 7")

	#PCA = 3, KNN, k = 10
	cross_validate(1, 10, 1, 3, "PCA = 3, KNN, k = 10")

	#PCA 5, KNN, k = 1
	cross_validate(1, 1, 1, 5, "PCA = 5, KNN, k = 1")

	#PCA = 5, KNN, k = 3
	cross_validate(1, 3, 1, 5, "PCA = 5, KNN, k = 3")

	#PCA = 5, KNN, k = 5
	cross_validate(1, 5, 1, 5, "PCA = 5, KNN, k = 5")

	#PCA = 5, KNN, k = 7
	cross_validate(1, 7, 1, 5, "PCA = 5, KNN, k = 7")

	#PCA = 5, KNN, k = 10
	cross_validate(1, 10, 1, 5, "PCA = 5, KNN, k = 10")

	#PCA 7, KNN, k = 1
	cross_validate(1, 1, 1, 7, "PCA = 7, KNN, k = 1")

	#PCA = 7, KNN, k = 3
	cross_validate(1, 3, 1, 7, "PCA = 7, KNN, k = 3")

	#PCA = 7, KNN, k = 5
	cross_validate(1, 5, 1, 7, "PCA = 7, KNN, k = 5")

	#PCA = 7, KNN, k = 7
	cross_validate(1, 7, 1, 7, "PCA = 7, KNN, k = 7")

	#PCA = 7, KNN, k = 10
	cross_validate(1, 10, 1, 7, "PCA = 7, KNN, k = 10")

	#visualize features
	#visualize_features.plot_1d(X_train,Y_train)
	'''