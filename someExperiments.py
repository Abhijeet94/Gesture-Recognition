import os
import csv
import numpy as np

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
	zeta[T-1, ;, :] = 1

	# Compute
	for t in range(T - 1):
		alphaA = np.multiply(alpha[t, :].reshape(N, 1), A)
		BBeta = np.multiply(B[:, Ob[t+1]], beta[t+1, :])
		zeta[t, :, :] = np.multiply(alphaA, BBeta)
		zeta[t, :, :] = zeta[t, :, :]/(np.sum(zeta[t, :, :]))	

	return zeta

def computeNewPi(gammaArray):
	return np.sum(gammaArray[:, 0, :], axis=0).reshape(N)/gammaArray.shape[0]

def computeNewA(gammaArray, zetaArray):
	numerator = np.sum(zetaArray, axis=(0,1))
	denominator = np.sum(numerator, axis=1)
	return numerator/denominator

def computeNewB(gammaArray, ObArray):
	numerator = np.sum() # Complete this
	denominator = np.sum(numerator, axis=1)
	return numerator/denominator

def getLogLikelihood(pi, A, B):
	pass

def BaumWelch(ObservationArray):

	# Initialize model parameters
	A = np.random.rand(N, N) + 0.001
	B = np.random.rand(N, M) + 0.001
	pi = np.random.rand(N) + 0.001
	ll_old = 0
	threshold = 1e-5

	while True:

		gammaArray = np.zeros((ObservationArray.shape[0]), T, N)
		zetaArray = np.zeros((ObservationArray.shape[0]), T, N, N)

		# Expectation step
		for i, example in enumerate(ObservationArray):
			alpha, beta, gammaArray[i] = forwardBackward(pi, A, B, ObservationArray[i])
			zetaArray[i] = computeZeta(A, B, alpha, beta, ObservationArray[i])

		# Maximization step
		pi = computeNewPi(gammaArray)
		A = computeNewA(gammaArray, zetaArray)
		B = computeNewB(gammaArray, ObservationArray)

		# Evaluate log-likelihood
		ll_new = getLogLikelihood(pi, A, B)

		# break if low change
		if abs(ll_new - ll_old) < threshold:
			break
		else:
			ll_old = ll_new

	# return final model parameters
	return pi, A, B

def trainHMMmodels():
	trainedModels = [None] * len(gestures)
	fileList = getAllFilesInFolder(DATA_FOLDER)
	clusterCenters = getClusterCenters()

	for i, g in enumerate(gestures):

		observationList = []
		maxT = 0

		for f in fileList:
			if f.startswith(g):
				dataInFile = dataToNpArray(os.path.join(DATA_FOLDER, f))
				discretizedData = assignPointsToNearestCluster(dataInFile[1:7], clusterCenters)
				observationList.append(discretizedData.reshape(discretizedData.size))

				if maxT < dataInFile.shape[0]:
					maxT = dataInFile.shape[0]

		observationArray = np.zeros((len(observationList), maxT))
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
	cl, ce = runKMeans(10, dataToNpArray(os.path.join(DATA_FOLDER, filename))[:, 1:7])
	print cl
	print ce