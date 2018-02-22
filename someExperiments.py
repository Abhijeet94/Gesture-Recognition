import os
import csv
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

def getClusterCenters():
	data = None
	fileList = getAllFilesInFolder(DATA_FOLDER)
	for f in fileList:
		if data is None:
			data = dataToNpArray(os.path.join(DATA_FOLDER, filename))[:, 1:7]
		else:
			data = np.vstack((data, dataToNpArray(os.path.join(DATA_FOLDER, filename))[:, 1:7]))
	
	clusters, clusterCenters = runKMeans(M, data)
	return clusterCenters

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
	for t in range(T - 1):
		for i in range(N):
			logAlpha[t+1, i] = logsumexp(logAlpha[t, :] + np.log(A[:, i]))
		logAlpha[t+1, :] = logAlpha[t+1, :] + np.log(B[:, Ob[t+1]-1])

	# Backward pass
	for t in reversed(range(T-1)):
		for i in range(N):
			logBeta[t, i] = logsumexp(np.log(A[i, :]) + np.log(B[:, Ob[t+1]-1]) + logBeta[t+1, :])

	# Compute logGamma (posterior probability of hidden state)
	logGamma = np.add(logAlpha, logBeta)
	logGamma = np.subtract(logGamma, logsumexp(logGamma, axis=1).reshape(logGamma.shape[0], 1))

	return logAlpha, logBeta, logGamma

def computeLogZeta(A, B, logAlpha, logBeta, Ob):
	TT = Ob.size
	T = np.count_nonzero(Ob)
	logZeta = np.zeros((TT, N, N))

	# Initialize
	logZeta = np.tile(logAlpha.reshape(logAlpha.shape[0], logAlpha.shape[1], 1), (1, 1, N))
	logZeta = logZeta + np.tile(logBeta.reshape(logBeta.shape[0], 1, logBeta.shape[1]), (1, N, 1))	
	logZeta = logZeta + np.tile(np.log(A).reshape(1, A.shape[0], A.shape[1]), (TT, 1, 1))	
	logZeta = logZeta + np.tile(np.log(np.transpose(B[:, Ob - 1])).reshape(logBeta.shape[0], 1, logBeta.shape[1]), (1, N, 1))	
	logZeta = np.subtract(logZeta, np.tile(logsumexp(logZeta, axis=(1, 2)).reshape(TT, 1, 1), (1, N, N)))

	return logZeta

def computeNewPi(logGammaArray):
	return np.exp(logsumexp(logGammaArray[:, 0, :], axis=0).reshape(N))/logGammaArray.shape[0]

def computeNewA(logGammaArray, logZetaArray):
	numerator = np.exp(logsumexp(logZetaArray, axis=(0,1)))
	denominator = np.sum(numerator, axis=1)
	return numerator/denominator

def computeNewB(logGammaArray, ObArray):
	numerator = np.zeros((N, M))
	for x in range(M):
		x = x + 1
		replicatedObArray = np.tile((ObArray == x).reshape(ObArray.shape[0], ObArray.shape[1], 1), (1, 1, N))
		numerator[:, x-1] = np.exp(logsumexp(np.multiply(logGammaArray, replicatedObArray), axis=(0,1)))

	denominator = np.sum(numerator, axis=1)
	return numerator/denominator[:,None]

def getLogLikelihood(pi, A, B):
	return 0

########################################################################################################

def BaumWelch(ObservationArray):

	# Initialize model parameters
	A = np.random.rand(N, N) + 0.001
	A = A / np.sum(A, axis=1)
	B = np.random.rand(N, M) + 0.001
	B = B / np.sum(B, axis=1).reshape(N, 1)
	pi = np.random.rand(N) + 0.001
	pi = pi / np.sum(pi)

	ll_old = 0
	threshold = 1e-5
	numIterations = 0
	maxIterations = 100
	TT = ObservationArray.shape[1]

	while numIterations < maxIterations:

		logGammaArray = np.zeros((ObservationArray.shape[0], TT, N))
		logZetaArray = np.zeros((ObservationArray.shape[0], TT, N, N))
		logAlphaT = np.zeros((ObservationArray.shape[0], N))

		# Expectation step
		for i, example in enumerate(ObservationArray):
			logAlpha, logBeta, logGammaArray[i] = logForwardBackward(pi, A, B, ObservationArray[i])
			logZetaArray[i] = computeLogZeta(A, B, logAlpha, logBeta, ObservationArray[i])

			logAlphaT[i] = logAlpha[np.count_nonzero(ObservationArray[i]) - 1, :]

		# Maximization step
		pi = computeNewPi(logGammaArray)
		A = computeNewA(logGammaArray, logZetaArray)
		B = computeNewB(logGammaArray, ObservationArray)

		# Evaluate log-likelihood
		# ll_new = getLogLikelihood(pi, A, B)
		ll_new = np.sum(np.exp(logsumexp(logAlphaT, axis=1))) / ObservationArray.shape[0]
		print 'Log-likelihood at iteration ' + str(numIterations) + ' is ' + str(ll_new)

		# break if low change
		if abs(ll_new - ll_old) < threshold:
			pass#break
		else:
			ll_old = ll_new

		numIterations = numIterations + 1

	# return final model parameters
	return pi, A, B

########################################################################################################

def trainHMMmodels():
	trainedModels = [None] * len(gestures)
	fileList = getAllFilesInFolder(DATA_FOLDER)
	# clusterCenters = getClusterCenters()
	kmeans = kMeans()
	# print np.unique(kmeans.labels_)
	print 'K-means done.'

	for i, g in enumerate(gestures):

		observationList = []
		maxT = 0

		for f in fileList:
			if f.startswith(g):
				dataInFile = dataToNpArray(os.path.join(DATA_FOLDER, f))
				# discretizedData = assignPointsToNearestCluster(dataInFile[1:7], clusterCenters)
				discretizedData = kmeans.predict(dataInFile[:, 1:7]) + 1
				# print np.unique(discretizedData)
				observationList.append(discretizedData.reshape(discretizedData.size))

				if maxT < dataInFile.shape[0]:
					maxT = dataInFile.shape[0]

		observationArray = np.zeros((len(observationList), maxT), dtype=int)
		for i, o in enumerate(observationList):
			observationArray[i, 0:(o.size)] = o

		pi, A, B = BaumWelch(observationArray)
		trainedModels[i] = (pi, A, B)

	return clusterCenters, trainedModels

def predict(trainedModels, x):
	threshold = 0.6
	bestTillNow = -1
	bestScore = -1
	for i, t in enumerate(trainedModels):
		p = calObservationPr(t, x)

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

	# seeOrientation(filename)

	# cl, ce = runKMeans(10, dataToNpArray(os.path.join(DATA_FOLDER, filename))[:, 1:7])
	# print cl
	# print ce

	trainHMMmodels()