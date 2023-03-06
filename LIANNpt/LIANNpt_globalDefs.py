"""LIANNpt_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see LIANNpt_main.py

# Usage:
see LIANNpt_main.py

# Description:
LIANNpt globalDefs

"""

import torch
useLovelyTensors = False
if(useLovelyTensors):
	import lovely_tensors as lt
	lt.monkey_patch()
else:
	torch.set_printoptions(profile="full")


useAlgorithmSMANN = True

stateTrainDataset = True
stateTestDataset = True


SMANNuseSoftmax = True	#required for useAlgorithmSMANN (disable for performance comparison with standard neural net/relu)
if(SMANNuseSoftmax):
	usePositiveWeights = True	#required
	if(usePositiveWeights):
		usePositiveWeightsClampModel = True	#clamp entire model weights to be positive (rather than per layer); currently required
	useInbuiltCrossEntropyLossFunction = True	#optional
else:
	usePositiveWeights = False	#required
	useInbuiltCrossEntropyLossFunction = True	#required

datasetShuffle = False	#optional

datasetName = 'blog-feedback'
#datasetName = 'tabular-benchmark'
if(datasetName == 'tabular-benchmark'):
	datasetNameFull = 'inria-soda/tabular-benchmark'
	classFieldName = 'class'
	trainFileName = 'clf_cat/albert.csv'
	testFileName = 'clf_cat/albert.csv'
	datasetShuffle = False	#backprop does not optimise with shuffled dataset
elif(datasetName == 'blog-feedback'):
	datasetNameFull = 'wwydmanski/blog-feedback'
	classFieldName = 'target'
	trainFileName = 'train.csv'
	testFileName = 'test.csv'

debugSmallNetwork = False
if(debugSmallNetwork):
	batchSize = 2
	numberOfLayers = 4
	hiddenLayerSize = 5	
	trainNumberOfEpochs = 1	#default: 10	#number of epochs to train
else:
	batchSize = 64
	numberOfLayers = 4
	hiddenLayerSize = 100
	trainNumberOfEpochs = 10	#default: 10	#number of epochs to train

printAccuracyRunningAverage = True
if(printAccuracyRunningAverage):
	runningAverageBatches = 10


learningRate = 0.005	#0.005	#0.0001

useLinearSublayers = True	#use multiple independent sublayers per linear layer
if(useLinearSublayers):
	linearSublayersNumber = 10
else:
	linearSublayersNumber = 1
	

relativeFolderLocations = False
userName = 'user'	#default: user
tokenString = "INSERT_HUGGINGFACE_TOKEN_HERE"	#default: INSERT_HUGGINGFACE_TOKEN_HERE

modelSaveNumberOfBatches = 100	#resave model after x training batches

dataDrive = '/datasets/'
workingDrive = '/large/source/ANNpython/LIANNpt/'

dataFolderName = 'data'
modelFolderName = 'model'
if(relativeFolderLocations):
	dataPathName = dataFolderName
	modelPathName = modelFolderName
else:	
	dataPathName = '/media/' + userName + dataDrive + dataFolderName
	modelPathName = '/media/' + userName + workingDrive + modelFolderName

def getModelPathNameFull(modelPathNameBase, modelName):
	modelPathNameFull = modelPathNameBase + '/' + modelName + '.pt'
	return modelPathNameFull
	
modelPathNameBase = modelPathName
modelName = 'modelLIANN'
modelPathNameFull = getModelPathNameFull(modelPathNameBase, modelName)
	
def printCUDAmemory(tag):
	print(tag)
	
	pynvml.nvmlInit()
	h = pynvml.nvmlDeviceGetHandleByIndex(0)
	info = pynvml.nvmlDeviceGetMemoryInfo(h)
	total_memory = info.total
	memory_free = info.free
	memory_allocated = info.used
	'''
	total_memory = torch.cuda.get_device_properties(0).total_memory
	memory_reserved = torch.cuda.memory_reserved(0)
	memory_allocated = torch.cuda.memory_allocated(0)
	memory_free = memory_reserved-memory_allocated  # free inside reserved
	'''
	print("CUDA total_memory = ", total_memory)
	#print("CUDA memory_reserved = ", memory_reserved)
	print("CUDA memory_allocated = ", memory_allocated)
	print("CUDA memory_free = ", memory_free)

def printe(str):
	print(str)
	exit()
