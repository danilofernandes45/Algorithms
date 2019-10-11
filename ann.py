from random import random, choice
from math import exp
import pandas as pd
import numpy as np

LEARN_RATE = 0.1

def generateNeuralNet(numLayers, lenLayer, lenInput):

	ann = []

	layer = []
	for j in range(lenLayer):
		layer.append( [ 0, [ choice([-1, 1])*random() for k in range(lenInput) ], 1 ] )
	ann.append(layer)

	for i in range(1, numLayers+1):
		layer = []
		for j in range(lenLayer):
			layer.append( [ 0, [ choice([-1, 1])*random() for k in range(lenLayer) ], 1 ] )
		ann.append(layer)

	ann.append( [ [ 0, [ choice([-1, 1])*random() for k in range(lenLayer) ], 1 ] ] )

	return ann

def sigm(x):
	try:
		e = exp(-x)
	except OverflowError:
		e = float("inf")
	except:
		e = 0
	finally:
		return ( 1 / ( 1 + e ) )

def relu(x):
	return max(0, x)

def dif_relu(x):
	return 1 if x > 0 else 0

def forward(ann, values, numLayers, lenLayer, lenInput):

	for j in range(lenLayer):

		summation = 0

		for k in range(lenInput):
			summation += ann[0][j][1][k] * values[k]

		ann[0][j][0] = relu(summation)

	for i in range(1, numLayers+1):
		for j in range(lenLayer):

			summation = 0

			for k in range(lenLayer):
				summation += ann[i][j][1][k] * ann[i-1][k][0]

			ann[i][j][0] = relu(summation)


	summation = 0

	for k in range(lenLayer):

		summation += ann[numLayers+1][0][1][k] * ann[numLayers][k][0]

	ann[numLayers+1][0][0] = sigm(summation)
	# print("Sum: %.2f"%summation)
	# print("Sigmoid: %.2f"%sigm(summation))
	#print("Return - %.5f"%ann[numLayers][0][0])

	return ann

def backward(ann, values, expecValue, numLayers, lenLayer, lenInput):

	difError = 2*( ann[numLayers+1][0][0] - expecValue ) * ( 1 - ann[numLayers+1][0][0] ) * ann[numLayers+1][0][0]
	ann[numLayers+1][0][2] = difError
	#print(difError)

	for i in range(lenLayer):
		ann[numLayers+1][0][1][i] -= LEARN_RATE * difError * ann[numLayers][i][0]

	for j in range(numLayers, 0, -1):

		for i in range(lenLayer):

			sum_dif = 0
			for u in ann[j+1]:
				sum_dif += u[2] * ( u[1][i] + LEARN_RATE * u[2] * ann[j][i][0] ) #Ocasiona erro numérico

			difError = sum_dif * dif_relu( ann[j][i][0] )
			ann[j][i][2] = difError

			for k in range(lenLayer):
				ann[j][i][1][k] -= LEARN_RATE * difError * ann[j-1][k][0]

	for i in range(lenLayer):

		sum_dif = 0
		for u in ann[1]:
			sum_dif += u[2] * ( u[1][i] + LEARN_RATE * u[2] * ann[0][i][0] ) #Ocasiona erro numérico

		difError = sum_dif * dif_relu( ann[0][i][0] )
		ann[0][i][2] = difError

		for k in range(lenInput):
			ann[0][i][1][k] -= LEARN_RATE * difError * values[k]

	return ann


def train(ann, numLayers, lenLayer, trainData, lenInput):

	for i in range(trainData.shape[0]):
		ann = forward(ann, trainData.iloc[i, 0:lenInput], numLayers, lenLayer, lenInput)
		ann = backward(ann, trainData.iloc[i, 0:lenInput], trainData.iloc[i, lenInput], numLayers, lenLayer, lenInput)

	return ann

def test(ann, numLayers, lenLayer, testData, lenInput):

	medium_abs_error = 0

	for i in range(testData.shape[0]):
		ann = forward(ann, testData.iloc[i, 0:lenInput], numLayers, lenLayer, lenInput)
		print("Class - %.5f"%testData.iloc[i, lenInput])
		print("Return - %.5f"%ann[numLayers+1][0][0])
		medium_abs_error += abs( testData.iloc[i, lenInput] - ann[numLayers+1][0][0] )

	medium_abs_error /= testData.shape[0]
	return medium_abs_error


#Data

# data = pd.read_csv("dataset/breast-cancer.csv")
data = pd.read_csv("dataset/breast-cancer.csv")

numLayers = 30
lenLayer = 7
lenInput = data.shape[1] - 1

ann = generateNeuralNet(numLayers, lenLayer, lenInput)
print(ann[numLayers+1][0][1])

#Train
ann = train(ann, numLayers, lenLayer, data.iloc[ 0 : 2 * data.shape[0] // 3], lenInput)

print(test(ann, numLayers, lenLayer, data.iloc[2 * data.shape[0] // 3 : ], lenInput))

print(ann[numLayers+1][0])
print(forward(ann, [-1]*9, numLayers, lenLayer, 9)[numLayers+1][0][0])
print(forward(ann, [0.1]*9, numLayers, lenLayer, 9)[numLayers+1][0][0])
print(forward(ann, [0.9]*9, numLayers, lenLayer, 9)[numLayers+1][0][0])
