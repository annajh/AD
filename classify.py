import data
import SVM
import visualize_features
from sklearn.mixture import GMM
import numpy as np

if __name__ == "__main__":
	X, Y, subjectID = data.load_data("control_features_combinedSubject.txt", "dementia_features_combinedSubject.txt")
	# get useful features
	X = data.get_useful_features_mat(X)

	X_testing = np.array([[1.22354785408e-92, -1.,  2.],
                    	[ 3.78725178547e-43,  0.,  0.],
                    	[ 1.50844307291e-66,  1., -1.]])
	X_testing_2 = data.normalize_features(X_testing)
	print X_testing_2

	#normalize features
	print X
	X_scaled = data.normalize_features(X)
	print X_scaled

	# split training and testing data
	X_train, Y_train, X_test, Y_test, trainID, testID = data.split_train_test(X_scaled,Y,subjectID)

	#SVM
	clf = SVM.train(X_train,Y_train)
	#print X_train, Y_train
	#print clf
	#print SVM.test(X_test,clf)
	#print Y_test
	#SVM.plot(clf,X_train,Y_train)

	visualize_features.plot_1d(X_train,Y_train)
	#GMM
	# Add parameters as needed
	#comp_range = [1,2,3,5]  # hyperparameter, try different number of mixture components
	#covar_type = 'spherical'  # the only type I've learned (or remembered learning) tbh
	#model = GMM(n_components=1, covariance_type=covar_type)
	#model.fit(X_control)

