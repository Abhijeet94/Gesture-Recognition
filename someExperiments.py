import os
import csv
import pdb
import math
import numpy as np
from scipy.misc import logsumexp
from sklearn.cluster import KMeans

import sys
sys.path.insert(0, 'KalmanFilter')
from kalmanFilter2 import UKF
from kalmanFilter import checkGyroIntegration
from plots import plotPredictions

from kmeans import *

DATA_FOLDER = 'train_data'
gestures = ['beat3', 'beat4', 'circle', 'eight', 'inf', 'wave']

N = 10 # Number of hidden states
M = 30 # Number of observation classes
# N = 4 # Dummy
# M = 5 # Dummy

########################################################################################################

def getAllFilesInFolder(folder):
	fileList = []
	for fName in os.listdir(folder):
		if os.path.isfile(os.path.join(folder, fName)):
			fileList.append(fName)
	return fileList

def dataToNpArray(filename):
	fileData = []
	with open(filename) as tsv:
		for line in csv.reader(tsv, delimiter="\t"):
			fileData.append(line)
	return np.array(fileData, dtype=np.float64)

def seeOrientation(filename):
	sensorData = dataToNpArray(os.path.join(DATA_FOLDER, filename))
	numInstances = sensorData.shape[0]
	timestamps = sensorData[:, 0]
	timestamps = timestamps.reshape(1, numInstances) / 1000
	gyroData = sensorData[:, 4:7].reshape(numInstances, 3)
	accelerometerData = sensorData[:, 1:4].reshape(numInstances, 3)

	# filterResult = UKF(gyroData, accelerometerData, timestamps)
	filterResult = checkGyroIntegration(gyroData, timestamps)
	plotPredictions(filterResult, timestamps)

def kMeans():
	data = None
	fileList = getAllFilesInFolder(DATA_FOLDER)
	for f in fileList:
		if data is None:
			data = dataToNpArray(os.path.join(DATA_FOLDER, filename))[:, 1:7]
		else:
			data = np.vstack((data, dataToNpArray(os.path.join(DATA_FOLDER, filename))[:, 1:7]))
	
	kmeans = KMeans(n_clusters=M).fit(data)
	return kmeans

########################################################################################################
# Dummy data to check algorithm

def generate_observations(model_name, T):
	"""
	The Ride model from west Philly to Engineering.
	State : Chesnut St., Walnut St., Spruce St., Pine St. 
	Observation : Students (five - S, W, P, W, C) 
	model_name : name of a model
	T : length of a observation sequence to generate
	"""
	if model_name == 'oober':
		A = np.array([[0.4, 0.4, 0.1, 0.1],
						[0.3, 0.3, 0.3, 0.1],
						[0.1, 0.3, 0.3, 0.3],
						[0.1, 0.1, 0.4, 0.4]], dtype=np.float32)

		B = np.array([[0.0, 0.2, 0.0, 0.2, 0.6],
						[0.1, 0.4, 0.0, 0.4, 0.1],
						[0.5, 0.2, 0.1, 0.2, 0.0],
						[0.3, 0.1, 0.5, 0.1, 0.0]], dtype=np.float32)

		Pi = np.array([0.3, 0.4, 0.1, 0.2], dtype=np.float32)

	elif model_name == 'nowaymo':
		A = np.array([[0.5, 0.1, 0.1, 0.3],
						[0.2, 0.6, 0.1, 0.1],
						[0.05, 0.1, 0.8, 0.05],
						[0, 0.1, 0.2, 0.7]], dtype=np.float32)

		B = np.array([[0.0, 0.2, 0.0, 0.2, 0.6],
						[0.1, 0.4, 0.0, 0.4, 0.1],
						[0.5, 0.2, 0.1, 0.2, 0.0],
						[0.3, 0.1, 0.5, 0.1, 0.0]], dtype=np.float32)

		Pi = np.array([0.2, 0.2, 0.1, 0.5], dtype=np.float32)

	elif model_name == 'dummy':
		A = np.array([[0.0, 1.0, 0.0, 0.0],
						[0.0, 0.0, 1.0, 0.0],
						[0.0, 0.0, 0.0, 1.0],
						[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

		B = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
						[0.0, 1.0, 0.0, 0.0, 0.0],
						[0.0, 0.0, 1.0, 0.0, 0.0],
						[0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32)

		Pi = np.array([0.25,0.25,0.25,0.25], dtype=np.float32)

	state = inv_sampling(Pi)
	obs_sequence = []
	for t in xrange(T):
		obs_sequence.append(inv_sampling(B[state,:]))
		state = inv_sampling(A[state,:])
	return np.array(obs_sequence)


def inv_sampling(pdf):
	r = np.random.rand() 
	for (i,p) in enumerate(np.cumsum(pdf)):
		if r <= p:
			return i

def trainHMMmodelsWithDummyData():
	observationArray = np.zeros((15, 20), dtype=int)
	observationArray[0, 0:17] = generate_observations('oober', 17) + 1
	observationArray[1, 0:15] = generate_observations('oober', 15) + 1
	observationArray[2, 0:20] = generate_observations('oober', 20) + 1
	observationArray[3, 0:20] = generate_observations('oober', 20) + 1
	observationArray[4, 0:19] = generate_observations('oober', 19) + 1
	observationArray[5, 0:17] = generate_observations('oober', 17) + 1
	observationArray[6, 0:15] = generate_observations('oober', 15) + 1
	observationArray[7, 0:20] = generate_observations('oober', 20) + 1
	observationArray[8, 0:20] = generate_observations('oober', 20) + 1
	observationArray[9, 0:19] = generate_observations('oober', 19) + 1
	observationArray[10, 0:17] = generate_observations('oober', 17) + 1
	observationArray[11, 0:15] = generate_observations('oober', 15) + 1
	observationArray[12, 0:20] = generate_observations('oober', 20) + 1
	observationArray[13, 0:20] = generate_observations('oober', 20) + 1
	observationArray[14, 0:19] = generate_observations('oober', 19) + 1

	pi, A, B = BaumWelch(observationArray)

	np.set_printoptions(precision=2)
	print 'pi\n' + str(pi)
	print 'A\n' + str(A)
	print 'B\n' + str(B)

	observationArray2 = np.zeros((15, 20), dtype=int)
	observationArray2[0, 0:17] = generate_observations('nowaymo', 17) + 1
	observationArray2[1, 0:15] = generate_observations('nowaymo', 15) + 1
	observationArray2[2, 0:20] = generate_observations('nowaymo', 20) + 1
	observationArray2[3, 0:20] = generate_observations('nowaymo', 20) + 1
	observationArray2[4, 0:19] = generate_observations('nowaymo', 19) + 1
	observationArray2[5, 0:17] = generate_observations('nowaymo', 17) + 1
	observationArray2[6, 0:15] = generate_observations('nowaymo', 15) + 1
	observationArray2[7, 0:20] = generate_observations('nowaymo', 20) + 1
	observationArray2[8, 0:20] = generate_observations('nowaymo', 20) + 1
	observationArray2[9, 0:19] = generate_observations('nowaymo', 19) + 1
	observationArray2[10, 0:17] = generate_observations('nowaymo', 17) + 1
	observationArray2[11, 0:15] = generate_observations('nowaymo', 15) + 1
	observationArray2[12, 0:20] = generate_observations('nowaymo', 20) + 1
	observationArray2[13, 0:20] = generate_observations('nowaymo', 20) + 1
	observationArray2[14, 0:19] = generate_observations('nowaymo', 19) + 1

	pi2, A2, B2 = BaumWelch(observationArray2)

	np.set_printoptions(precision=2)
	print 'pi\n' + str(pi2)
	print 'A\n' + str(A2)
	print 'B\n' + str(B2)

	somedata = generate_observations('nowaymo', 19) + 1
	print calObservationPr((pi2, A2, B2), somedata)
	print calObservationPr((pi, A, B), somedata)

	somedata = generate_observations('nowaymo', 19) + 1
	print calObservationPr((pi2, A2, B2), somedata)
	print calObservationPr((pi, A, B), somedata)

	somedata = generate_observations('nowaymo', 19) + 1
	print calObservationPr((pi2, A2, B2), somedata)
	print calObservationPr((pi, A, B), somedata)

	somedata = generate_observations('oober', 19) + 1
	print calObservationPr((pi2, A2, B2), somedata)
	print calObservationPr((pi, A, B), somedata)

	somedata = generate_observations('oober', 19) + 1
	print calObservationPr((pi2, A2, B2), somedata)
	print calObservationPr((pi, A, B), somedata)

# Dummy data to check algorithm
########################################################################################################

# Not in log space
# For reference 

def forwardBackward(pi, A, B, Ob):
	pi = pi.reshape(N)
	B = B.reshape(N, M)
	TT = Ob.size
	T = np.count_nonzero(Ob)
	Ob = Ob.reshape(TT)

	alpha = np.zeros((TT, N))
	beta = np.zeros((TT, N))
	gamma = np.zeros((TT, N))

	# Initialize
	alpha[0, :] = np.multiply(pi, B[:, Ob[0]])
	beta[T-1, :] = 1

	# Forward pass
	for t in range(T - 1):
		alpha[t+1, :] = np.multiply(np.matmul(alpha[t, :], A), B[:, Ob[t+1]])

	# Backward pass
	for t in reversed(range(T-1)):
		beta[t, :] = np.matmul(A, np.multiply(B[:, Ob[t+1]], beta[t+1, :]))

	# Compute gamma (posterior probability of hidden state)
	gamma = np.multiply(alpha, beta)
	gamma = np.divide(gamma, np.sum(gamma, axis=1))

	return alpha, beta, gamma

def computeZeta(A, B, alpha, beta, Ob):
	TT = Ob.size
	T = np.count_nonzero(Ob)
	zeta = np.zeros((TT, N, N))

	# Initialize
	zeta[T-1, :, :] = 1

	# Compute
	for t in range(T - 1):
		alphaA = np.multiply(alpha[t, :].reshape(N, 1), A)
		BBeta = np.multiply(B[:, Ob[t+1]], beta[t+1, :])
		zeta[t, :, :] = np.multiply(alphaA, BBeta)
		zeta[t, :, :] = zeta[t, :, :]/(np.sum(zeta[t, :, :]))	

	return zeta

def computeNewPiOld(gammaArray):
	return np.sum(gammaArray[:, 0, :], axis=0).reshape(N)/gammaArray.shape[0]

def computeNewAOld(gammaArray, zetaArray):
	numerator = np.sum(zetaArray, axis=(0,1))
	denominator = np.sum(numerator, axis=1)
	return numerator/denominator

def computeNewBOld(gammaArray, ObArray):
	numerator = np.sum() # Complete this
	denominator = np.sum(numerator, axis=1)
	return numerator/denominator

# Not in log space
# For reference 
########################################################################################################

def logForwardPass(logAlpha, A, B, Ob):
	T = np.count_nonzero(Ob)

	for t in range(T - 1):
		logAlpha[t+1, :] = logsumexp(np.add(logAlpha[t, :].reshape(N, 1), np.log(A)), axis=0)
		logAlpha[t+1, :] = logAlpha[t+1, :] + np.log(B[:, Ob[t+1]-1])

	return logAlpha

def logBackwardPass(logBeta, A, B, Ob):
	T = np.count_nonzero(Ob)

	for t in reversed(range(T-1)):
		logBeta[t, :] = logsumexp(np.log(np.transpose(A)) + np.log(B[:, Ob[t+1]-1]).reshape(N, 1) + logBeta[t+1, :].reshape(N, 1), axis=0)

	return logBeta

def logForwardBackward(pi, A, B, Ob):
	pi = pi.reshape(N)
	B = B.reshape(N, M)
	TT = Ob.size
	T = np.count_nonzero(Ob)
	Ob = Ob.reshape(TT)

	logAlpha = np.full((TT, N), -1 * np.inf)
	logBeta = np.full((TT, N), -1 * np.inf)
	logGamma = np.full((TT, N), -1 * np.inf)

	# Initialize
	logAlpha[0, :] = np.log(pi) + np.log(B[:, Ob[0]-1])
	logBeta[T-1, :] = math.log(1)

	# Forward pass
	logAlpha = logForwardPass(logAlpha, A, B, Ob)

	# Backward pass
	logBeta = logBackwardPass(logBeta, A, B, Ob)

	# Compute logGamma (posterior probability of hidden state)
	logGamma = np.add(logAlpha, logBeta)
	logGamma[0:T, :] = np.subtract(logGamma[0:T, :], logsumexp(logGamma[0:T, :], axis=1).reshape(T, 1))

	return logAlpha, logBeta, logGamma

def computeLogZeta(A, B, logAlpha, logBeta, Ob):
	TT = Ob.size
	T = np.count_nonzero(Ob)
	logZeta = np.full((TT, N, N), -1 * np.inf)

	# Initialize
	logZeta[0:T-1, :, :] = np.tile(logAlpha[0:T-1, :].reshape(T-1, N, 1), (1, 1, N))
	logZeta[0:T-1, :, :] = logZeta[0:T-1, :, :] + np.tile(np.log(A).reshape(1, N, N), (T-1, 1, 1))	

	logZeta[0:T-1, :, :] = logZeta[0:T-1, :, :] + np.tile(logBeta[1:T, :].reshape(T-1, 1, N), (1, N, 1))
	logZeta[0:T-1, :, :] = logZeta[0:T-1, :, :] + np.tile(np.log(np.transpose(B[:, Ob[1:T] - 1])).reshape(T-1, 1, N), (1, N, 1))

	logZeta[0:T-1, :, :] = np.subtract(logZeta[0:T-1, :, :], logsumexp(logZeta[0:T-1, :, :], axis=(1, 2))[:, None, None])

	return logZeta

def computeNewPi(logGammaArray):
	return np.exp(logsumexp(logGammaArray[:, 0, :], axis=0).reshape(N))/logGammaArray.shape[0]

def computeNewA(logGammaArray, logZetaArray, ObArray):
	E = ObArray.shape[0]

	numerator = np.zeros((N, N))
	for e in range(E):
		T = np.count_nonzero(ObArray[e])
		numerator = numerator + np.exp(logsumexp(logZetaArray[e, 0:T-1, :, :], axis=0))
	# print numerator

	denominator = np.zeros((N))
	for e in range(E):
		T = np.count_nonzero(ObArray[e])
		denominator = denominator + np.exp(logsumexp(logGammaArray[e, 0:T-1, :], axis=0))

	return numerator/denominator[:,None]

def computeNewB(logGammaArray, ObArray):
	E = ObArray.shape[0]

	numerator = np.full((N, M), -1 * np.inf)
	for i in range(N):
		for x in range(M):
			correct_t = (ObArray == (x+1))
			if np.count_nonzero(correct_t) > 0:
				numerator[i, x] = logsumexp(logGammaArray[correct_t, i])
	numerator = np.exp(numerator)

	denominator = np.zeros((N))
	for e in range(E):
		T = np.count_nonzero(ObArray[e])
		denominator = denominator + np.exp(logsumexp(logGammaArray[e, 0:T, :], axis=0))

	return numerator/denominator[:,None]

def calObservationPr(t, x):
	pi = t[0]
	A = t[1]
	B = t[2]
	Ob = x.reshape(x.size)
	TT = Ob.size

	logAlpha = np.full((TT, N), -1 * np.inf)
	# Initialize
	logAlpha[0, :] = np.log(pi) + np.log(B[:, Ob[0]-1])
	# Forward pass
	logAlpha = logForwardPass(logAlpha, A, B, Ob)

	logLikelihood = (logsumexp(logAlpha[TT-1, :]))

	return logLikelihood


########################################################################################################

def BaumWelch(ObservationArray):

	# Initialize model parameters
	A = np.random.rand(N, N) + 0.001
	A = A / np.sum(A, axis=1).reshape(N, 1)
	B = np.random.rand(N, M) + 0.001
	B = B / np.sum(B, axis=1).reshape(N, 1)
	pi = np.ones(N)
	pi = pi / np.sum(pi)

	ll_old = 0
	threshold = 1e-5
	iterCount = 0
	maxIterations = 25
	TT = ObservationArray.shape[1]
	E = ObservationArray.shape[0]

	while iterCount < maxIterations:

		logGammaArray = np.zeros((E, TT, N))
		logZetaArray = np.zeros((E, TT, N, N))
		logAlphaT = np.zeros((E, N))

		# Expectation step
		for i, example in enumerate(ObservationArray):
			logAlpha, logBeta, logGammaArray[i] = logForwardBackward(pi, A, B, ObservationArray[i])
			logZetaArray[i] = computeLogZeta(A, B, logAlpha, logBeta, ObservationArray[i])

			logAlphaT[i] = logAlpha[np.count_nonzero(ObservationArray[i]) - 1, :]

		# Maximization step
		pi = computeNewPi(logGammaArray)
		A = computeNewA(logGammaArray, logZetaArray, ObservationArray)
		# print np.sum(A, axis=1)
		B = computeNewB(logGammaArray, ObservationArray)
		# print np.sum(B, axis=1)

		# Evaluate log-likelihood
		ll_new = np.sum(np.exp(logsumexp(logAlphaT, axis=1))) / E
		print 'Log-likelihood at iteration ' + str(iterCount) + ' is ' + str(ll_new)

		# break if low change
		if abs(ll_new - ll_old) < threshold:
			pass#break
		else:
			ll_old = ll_new

		iterCount = iterCount + 1

	# return final model parameters
	return pi, A, B

########################################################################################################

def trainHMMmodels():
	trainedModels = [None] * len(gestures)
	fileList = getAllFilesInFolder(DATA_FOLDER)
	kmeans = kMeans()
	# print np.unique(kmeans.labels_)
	print 'K-means done.'

	for i, g in enumerate(gestures):

		observationList = []
		maxT = 0

		for f in fileList:
			if f.startswith(g):
				dataInFile = dataToNpArray(os.path.join(DATA_FOLDER, f))
				discretizedData = kmeans.predict(dataInFile[:, 1:7]) + 1
				# print np.unique(discretizedData)
				observationList.append(discretizedData.reshape(discretizedData.size))

				if maxT < discretizedData.size:
					maxT = discretizedData.size

		observationArray = np.zeros((len(observationList), maxT), dtype=int)
		for oi, o in enumerate(observationList):
			observationArray[oi, 0:(o.size)] = o

		pi, A, B = BaumWelch(observationArray)
		trainedModels[i] = (pi, A, B)

	return kmeans, trainedModels

def predict(trainedModels, data, kmeans):
	threshold = -np.inf
	bestTillNow = -np.inf
	bestScore = -np.inf

	x = kmeans.predict(data[:, 1:7]) + 1

	for i, t in enumerate(trainedModels):
		p = calObservationPr(t, x)
		print 'Score for ' + gestures[i] + ': ' + str(p)

		if p > bestScore:
			bestScore = p
			bestTillNow = i

	if bestScore > threshold:
		return gestures[bestTillNow]
	else:
		return 'Something else'

if __name__ == "__main__":
	filename = "inf11.txt"
	filename = "wave01.txt"

	# trainHMMmodelsWithDummyData()
	# exit()

	# seeOrientation(filename)

	kmeans, trainedModels = trainHMMmodels()

	filename = "wave01.txt"
	dataInTestFile = dataToNpArray(os.path.join(DATA_FOLDER, filename))
	print 'Prediction for ' + filename + ' is: ' + predict(trainedModels, dataInTestFile, kmeans)

	filename = "inf16.txt"
	dataInTestFile = dataToNpArray(os.path.join(DATA_FOLDER, filename))
	print 'Prediction for ' + filename + ' is: ' + predict(trainedModels, dataInTestFile, kmeans)

	filename = "circle18.txt"
	dataInTestFile = dataToNpArray(os.path.join(DATA_FOLDER, filename))
	print 'Prediction for ' + filename + ' is: ' + predict(trainedModels, dataInTestFile, kmeans)