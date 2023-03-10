"""LIANNpt_SMANNmodel.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see LIANNpt_main.py

# Usage:
see LIANNpt_main.py

# Description:
LIANNpt softmax artificial neural network (SMANN) model

"""

import torch as pt
from torch import nn
from ANNpt_globalDefs import *
from torchmetrics.classification import Accuracy
import ANNpt_linearSublayers


class SMANNconfig():
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
		
class SMANNmodel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		layersLinearList = []
		layersActivationList = []
		for layerIndex in range(config.numberOfLayers):
			linear = ANNpt_linearSublayers.generateLinearLayer(self, layerIndex, config)
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
		for layerIndex in range(self.config.numberOfLayers):
			x = ANNpt_linearSublayers.executeLinearLayer(self, layerIndex, x, self.layersLinear[layerIndex])
			if(debugSmallNetwork):
				print("layerIndex = ", layerIndex)
				print("x after linear = ", x)
			if(layerIndex == self.config.numberOfLayers-1):
				if(not useInbuiltCrossEntropyLossFunction):
					x = ANNpt_linearSublayers.executeActivationLayer(self, layerIndex, x, self.layersActivation[layerIndex])	#CHECKTHIS
					x = torch.log(x)
			else:
				x = ANNpt_linearSublayers.executeActivationLayer(self, layerIndex, x, self.layersActivation[layerIndex])
			if(debugSmallNetwork):
				print("x after activation = ", x)
		#print("x = ", x)
		#print("y = ", y)
		loss = self.lossFunction(x, y)
		accuracy = self.accuracyFunction(x, y)
		accuracy = accuracy.detach().cpu().numpy()
		
		return loss, accuracy


