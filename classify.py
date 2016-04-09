import numpy as np
from sklearn.mixture import GMM
from sklearn import svm
import random
import warnings

def load_data(control_file, dementia_file):
	"""
	Loads feature vectors from file into matrices X and Y
	"""
	X = []
	Y = []
	subjectID = []
	with open(control_file, 'r') as inf_control:
		for line in inf_control:
			features_str = line.split()
			features = map(float, features_str[1:])
			if len(features) == 0: continue  # in case there's empty lines in file
			X.append(features)
			Y.append(0)
			subjectID.append(features_str[0])

	with open(dementia_file, 'r') as inf_dementia:
		for line in inf_dementia:
			features_str = line.split()
			features = map(float, features_str[1:])
			if len(features) == 0: continue
			X.append(features)
			Y.append(1)
			subjectID.append(features_str[0])

	return np.array(X),Y, subjectID

#extract useful features from vector
def get_useful_features_vec(orig_vector):
	new_vector = []
	#0: TTR, 1: BI, 2: HS, 3: %pron, 4: %adj, 5: %PN, 6: %v
	for i in range(5,12):
		new_vector.append(orig_vector[i])
	#7: %filler words
	new_vector.append(orig_vector[13])
	#8: speech_rate, #9: articulation_rate, #10: asd, #11: mean_length_utterance
	#12: min_diff_f1f2, 13: max_diff_f1f2, 14: mean_diff_f1f2, 15: std_f1f2
	for i in range(19,27):
		new_vector.append(orig_vector[i])
	return new_vector

#extract the useful features from vectors of the matrix
def get_useful_features_mat(orig_mat):
	new_mat = []
	for item in orig_mat:
		new_mat.append(get_useful_features_vec(item))
	return np.array(new_mat)

#return training and testing data set according to trainID and test ID
#if testID is not provided, randomize one ID to be test, all others train
#if only trainID is provided, randomize on ID to be test, all others train. trainID DISREGARDED
#forced name argument for testID, trainID
def split_train_test(X,Y,subjectID,**kwargs):
	testID = kwargs.pop('testID',[])
	trainID = kwargs.pop('trainID',[])
	if kwargs:
		raise TypeError('Unexpected **kwargs: %r' %kwargs)
	#if passed in trainID only
	if testID == [] and trainID != []:
		trainID = []
		warnings.warn('testID is not provided. trainID is ignored')
	#no trainID or testID, randomize a subject to be tested
	if (testID == []):
		testID = random.sample(subjectID, 1)
	trainID = subjectID[:]
	for ID in testID:
		trainID.remove(ID)
	print trainID, testID
	#get the list of indices in X,Y,subjectID from trainID and testID
	trainIndex = []
	testIndex = []
	for train in trainID:
		trainIndex.append(subjectID.index(train))
	for test in testID:
		testIndex.append(subjectID.index(test))
	print trainIndex, testIndex
	#according to indices, create X_train, Y_train, X_test, Y_test
	X_train = []
	Y_train = []
	X_test = []
	Y_test = []
	for index in trainIndex:
		X_train.append(X[index])
		Y_train.append(Y[index])
	for index in testIndex:
		X_test.append(X[index])
		Y_test.append(Y[index])
	#return
	return np.array(X_train), Y_train, np.array(X_test), Y_test, trainID, testID

if __name__ == "__main__":
	X, Y, subjectID = load_data("control_features_combinedSubject.txt", "dementia_features_combinedSubject.txt")
	# get useful features
	X = get_useful_features_mat(X)

	# Add parameters as needed
	#comp_range = [1,2,3,5]  # hyperparameter, try different number of mixture components
	#covar_type = 'spherical'  # the only type I've learned (or remembered learning) tbh

	# split training and testing data
	split_train_test(X,Y,subjectID,trainID =subjectID)

	#model = GMM(n_components=1, covariance_type=covar_type)
	#model.fit(X_control)

