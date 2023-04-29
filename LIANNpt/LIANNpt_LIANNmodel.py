"""LIANNpt_LIANNmodel.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
LIANNpt local inhibitory artificial neural network model

"""

import torch as pt
from torch import nn
from ANNpt_globalDefs import *
from torchmetrics.classification import Accuracy
import ANNpt_linearSublayers


class LIANNconfig():
	def __init__(self, batchSize, numberOfLayers, hiddenLayerSize, inputLayerSize, outputLayerSize, linearSublayersNumber, numberOfFeatures, numberOfClasses, datasetSize, numberOfClassSamples):
		self.batchSize = batchSize
		self.numberOfLayers = numberOfLayers
		self.hiddenLayerSize = hiddenLayerSize
		self.inputLayerSize = inputLayerSize
		self.outputLayerSize = outputLayerSize
		self.linearSublayersNumber = linearSublayersNumber
		self.numberOfFeatures = numberOfFeatures
		self.numberOfClasses = numberOfClasses
		self.datasetSize = datasetSize		
		self.numberOfClassSamples = numberOfClassSamples
		
class LIANNmodel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		layersLinearList = []
		layersActivationList = []
		for layerIndex in range(config.numberOfLayers):
			linear = ANNpt_linearSublayers.generateLinearLayer(self, layerIndex, config, parallelStreams=useLUANNonly)
			layersLinearList.append(linear)
		for layerIndex in range(config.numberOfLayers):
			activation = ANNpt_linearSublayers.generateActivationLayer(self, layerIndex, config)
			layersActivationList.append(activation)
		self.layersLinear = nn.ModuleList(layersLinearList)
		self.layersActivation = nn.ModuleList(layersActivationList)
	
		if(useInbuiltCrossEntropyLossFunction):
			self.lossFunction = nn.CrossEntropyLoss()
		else:
			self.lossFunction = nn.NLLLoss()	#nn.CrossEntropyLoss == NLLLoss(log(softmax(x)))
		self.accuracyFunction = Accuracy(task="multiclass", num_classes=self.config.outputLayerSize, top_k=1)
		
		ANNpt_linearSublayers.weightsSetPositiveModel(self)
				
	def forward(self, trainOrTest, x, y, optim=None, l=None):
		if(useLUANNonly):
			x = x.unsqueeze(dim=1).repeat(1, self.config.linearSublayersNumber, 1)
		for layerIndex in range(self.config.numberOfLayers):
			xPrev = x
			if(trainLastLayerOnly):
				x = x.detach()
			x = ANNpt_linearSublayers.executeLinearLayer(self, layerIndex, x, self.layersLinear[layerIndex], parallelStreams=useLUANNonly)
			if(debugSmallNetwork):
				print("layerIndex = ", layerIndex)
				print("x after linear = ", x)
			if(layerIndex == self.config.numberOfLayers-1):
				if(not useInbuiltCrossEntropyLossFunction):
					x = ANNpt_linearSublayers.executeActivationLayer(self, layerIndex, x, self.layersActivation[layerIndex], parallelStreams=useLUANNonly)	#CHECKTHIS
					x = torch.log(x)
			else:
				if(simulatedDendriticBranches):
					x, xIndex = self.performTopK(x)
				if(trainOrTest and LIANNlocalLearning):
					self.trainWeightsLayer(layerIndex, x, xIndex, xPrev)
				x = ANNpt_linearSublayers.executeActivationLayer(self, layerIndex, x, self.layersActivation[layerIndex], parallelStreams=useLUANNonly)
			if(debugSmallNetwork):
				print("x after activation = ", x)
				 
		#print("x = ", x)
		#print("y = ", y)
		loss = self.lossFunction(x, y)
		accuracy = self.accuracyFunction(x, y)
		accuracy = accuracy.detach().cpu().numpy()
		
		return loss, accuracy

	def performTopK(self, x):
		xMax = pt.max(x, dim=1, keepdim=False)
		x = xMax.values
		xIndex = xMax.indices
		return x, xIndex
	
	def trainWeightsLayer(self, layerIndex, x, xIndex, xPrev):
		#print("xPrev = ", xPrev)
		xPrevActive = pt.greater(xPrev, LIANNlocalLearningNeuronActiveThreshold).float()
		xPrevInactive = pt.less(xPrev, LIANNlocalLearningNeuronActiveThreshold).float()
		xPrevInactiveBase = xPrevInactive
		if(LIANNlocalLearningBias):
			xPrevActive = pt.multiply(xPrev, xPrevActive)	#higher the activation, the more weight should be increased
			xPrevInactive = pt.multiply(xPrev, xPrevInactive)
			xPrevInactive = 1 - xPrevInactive	#lower the activation, the more weight should be decreased
		xPrevInactive = 0 - xPrevInactive
		xPrevInactive = pt.multiply(xPrevInactive, xPrevInactiveBase)	#ensure all active cells are set to zero
		#print("xPrevActive = ", xPrevActive)
		#print("xPrevInactive = ", xPrevInactive)
		xPrevWeightUpdate = pt.add(xPrevActive, xPrevInactive)
		xPrevWeightUpdate = xPrevWeightUpdate*LIANNlocalLearningRate
		xPrevWeightUpdate = xPrevWeightUpdate.unsqueeze(1)
		xPrevWeightUpdate = xPrevWeightUpdate.unsqueeze(-1)
		xPrevWeightUpdate = xPrevWeightUpdate.repeat(1, hiddenLayerSize, 1, 1)
		#print("xPrevWeightUpdate = ", xPrevWeightUpdate)
		layerWeights = self.layersLinear[layerIndex].segregatedLinear.weight
		layerWeights = pt.reshape(layerWeights, (linearSublayersNumber, hiddenLayerSize, layerWeights.shape[1], layerWeights.shape[2]))
		topKweightsList = []
		#TODO: require method to perform simultaneous indexing over hiddenLayerSize dimension;
		for i in range(hiddenLayerSize):
			topKweightsI = layerWeights[xIndex[:, i], i]
			topKweightsList.append(topKweightsI)
		topKweights = pt.stack(topKweightsList, dim=1)
		topKweights = pt.add(topKweights, xPrevWeightUpdate)
		topKweights = pt.mean(topKweights, dim=0)	#take mean weight adjustment over batch
		layerWeights = pt.reshape(layerWeights, (linearSublayersNumber*hiddenLayerSize, layerWeights.shape[2], layerWeights.shape[3]))
		self.layersLinear[layerIndex].segregatedLinear.weight = pt.nn.Parameter(layerWeights)

