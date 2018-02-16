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


def BaumWelch():
	pass

def trainHMMmodels():
	trainedModels = [None] * len(gestures)
	fileList = getAllFilesInFolder(DATA_FOLDER)

	for i, g in enumerate(gestures):
		for f in fileList:
			if f.startswith(g):
				pass

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