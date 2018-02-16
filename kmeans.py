import os
import csv
import random
import pdb
import numpy as np

def getRandomPoint(data):
	dimension = data.shape[1]
	point = np.zeros((1, dimension))
	for d in range(dimension):
		minVal = np.amin(data[:, d])
		maxVal = np.amax(data[:, d])
		point[0, d] = random.random() * (maxVal - minVal) + minVal
	return point

def getInitialCenters(dimension, numClusters, data):
	initialCenters = np.zeros((numClusters, dimension))

	# # Generate points randomly in range
	# for d in range(dimension):
	# 	minVal = np.amin(data[:, d])
	# 	maxVal = np.amax(data[:, d])
	# 	for c in range(numClusters):
	# 		initialCenters[c, d] = random.random() * (maxVal - minVal) + minVal

	# Choose one of the data points randomly
	for c in range(numClusters):
		initialCenters[c, :] = data[random.randint(0, data.shape[0]-1), :]

	return initialCenters

def runKMeans(numClusters, data)	:
	threshold = 1e-10

	dimension = data.shape[1]
	numPoints = data.shape[0]

	initialCenters = getInitialCenters(dimension, numClusters, data)
	currentCenters = initialCenters

	currentAssignment = np.zeros((numPoints, 1))

	while True:

		# Assign each point to its nearest center
		reshapedDataPoints = np.tile(data.reshape(numPoints, dimension, 1), (1, 1, numClusters))
		reshapedCenters = np.tile(np.transpose(currentCenters).reshape(1, dimension, numClusters), (numPoints, 1, 1))
		calNearest = reshapedDataPoints - reshapedCenters
		calNearest = np.square(calNearest)
		calNearest = np.sum(calNearest, axis=1)
		calNearest = np.sqrt(calNearest).reshape(numPoints, numClusters)
		currentAssignment = np.argmin(calNearest, axis=1).reshape(numPoints)

		# Calculate the new mean of each cluster
		newCenters = np.zeros((numClusters, dimension))
		flag = -1
		for c in range(numClusters):
			numAssigned = np.count_nonzero(currentAssignment == c)
			if numAssigned == 0:
				flag = c
				break
			newCenters[c, :] = np.sum(data[currentAssignment == c, :], axis=0)/(1.0 * numAssigned)

		if flag != -1:
			currentCenters[flag] = data[random.randint(0, numPoints-1), :]
			print 'reassigning',
			continue

		# Test convergence
		change = (newCenters - currentCenters)
		change = np.square(change)
		change = np.sum(change, axis=1)
		change = np.sqrt(change)
		change = np.sum(change, axis=0)
		if change < threshold:
			break

		# Else repeat
		currentCenters = newCenters

	return currentAssignment, newCenters