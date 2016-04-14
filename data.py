__author__ = 'jennytou'
import random
import warnings
import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, RandomizedPCA


#A class to raise error messages
class ValidationError(Exception):
	def _init_(self,message):
		self.message = message
	def _str_(selfself):
		return repr(self.message)

def load_testing_visit(dirAlz, dirCtr, visitname):
	X = []
	with open(dirAlz, 'r') as inf_control:
		for line in inf_control:
			features_str = line.split()
			if len(features_str) == 0: continue  # in case there's empty lines in file
			visit = features_str[0]
			features = map(float, features_str[1:])
			if visit != visitname: continue
			else: X.append(features)
	inf_control.close()

	with open(dirCtr, 'r') as inf_control:
		for line in inf_control:
			features_str = line.split()
			if len(features_str) == 0: continue  # in case there's empty lines in file
			#print features_str
			visit = features_str[0]
			features = map(float, features_str[1:])
			if visit != visitname: continue
			else: X.append(features)
	inf_control.close()

	return np.array(X)


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
	#print trainID
	#print testID
	for ID in testID:
		trainID.remove(ID)
	#print trainID, testID
	#get the list of indices in X,Y,subjectID from trainID and testID
	trainIndex = []
	testIndex = []
	for train in trainID:
		trainIndex.append(subjectID.index(train))
	for test in testID:
		testIndex.append(subjectID.index(test))
	#print trainIndex, testIndex
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
	#check training set, testing set is not empty
	try:
		if X_train == []:
			raise ValidationError('training data is empty!')
		if X_test == []:
			raise ValidationError('testing data is empty!')
	except ValidationError as e:
		print e.message
		sys.exit(1)
	#if more testing data than training data
	if len(X_test)> len(X_train):
		warnings.warn('training data size is less than testing data!')
	#return
	return np.array(X_train), Y_train, np.array(X_test), Y_test, trainID, testID

#normalize all feature vectors
def normalize_features(X):
	min_max_scaler = MinMaxScaler()
	normalizer = min_max_scaler.fit(X)
	X_train_minmax = normalizer.transform(X)
	return X_train_minmax, normalizer

#types:default,  randomized
def reduce_dimension(X, n, type = 'default'):
    if type == 'default':
        pca = PCA(n_components = n)
    elif type == 'randomized':
        pca = RandomizedPCA(n_components = n)
    else:
        raise TypeError('type can only be "default" or "randomized"')
    pca.fit(X)
    return pca, pca.explained_variance_ratio_