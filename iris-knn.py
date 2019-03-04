# KNN on IRIS

import csv
import random
import math
import operator
 
# Loading the dataset and splitting it into training and testing
def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open('Iris.csv','r') as file:
		lines = file.readlines()
		dataset = []
		for i in lines:
			dataset.append(i.rstrip('\n'))

	data = []
	for i in dataset[1:]:
		l = i.split(',')
		k = []
		for i in range(4):
			k.append(float(l[i]))
		k.append(l[-1])
		data.append(k)

	# split = 0.67
	for x in range(len(data)):
		for y in range(4):
			data[x][y] = float(data[x][y])
		if random.random() < split:
			trainingSet.append(data[x])
		else:
			testSet.append(data[x])




# Function to calculate the distance between points
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)
 
#Function to find the k nearest neighbour
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

# Gets the responses of the neighbours
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
 
# Funciton to calculate accuracy of KNN
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	
# Main body
def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.5 # Atleast 2/3 of dataset for training
	loadDataset('iris.csv', split, trainingSet, testSet)
	print('Train set: ', len(trainingSet))
	print('Test set: ', len(testSet))
	# generate predictions
	predictions=[]
	k = 2 # Number of neighbours
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		print('predicted=' ,result, ', actual=' ,testSet[x][-1])
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ', accuracy, '%')


if __name__ =='__main__':
    main()
