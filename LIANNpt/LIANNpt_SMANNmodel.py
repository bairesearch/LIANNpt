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
from LIANNpt_globalDefs import *
from torchmetrics.classification import Accuracy


class SMANNconfig():
	def __init__(self, batchSize, numberOfLayers, hiddenLayerSize, inputLayerSize, outputLayerSize, linearSublayersNumber, numberOfFeatures, numberOfClasses):
		self.batchSize = batchSize
		self.numberOfLayers = numberOfLayers
		self.hiddenLayerSize = hiddenLayerSize
		self.inputLayerSize = inputLayerSize
		self.outputLayerSize = outputLayerSize
		self.linearSublayersNumber = linearSublayersNumber
		self.numberOfFeatures = numberOfFeatures
		self.numberOfClasses = numberOfClasses
		
class SMANNmodel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		layersLinearList = []
		layersActivationList = []
		for layerIndex in range(config.numberOfLayers):
			linear = self.generateLinearLayer(layerIndex, config)
			layersLinearList.append(linear)
		for layerIndex in range(config.numberOfLayers):
			activation = self.generateActivationLayer(layerIndex, config)
			layersActivationList.append(activation)
		self.layersLinear = nn.ModuleList(layersLinearList)
		self.layersActivation = nn.ModuleList(layersActivationList)
	
		if(useInbuiltCrossEntropyLossFunction):
			self.lossFunction = nn.CrossEntropyLoss()
		else:
			self.lossFunction = nn.NLLLoss()	#nn.CrossEntropyLoss == NLLLoss(log(softmax(x)))
		self.accuracyFunction = Accuracy(task="multiclass", num_classes=self.config.outputLayerSize, top_k=1)
		
		self.weightsSetPositiveModel()
				
	def forward(self, x, y):
		for layerIndex in range(self.config.numberOfLayers):
			x = self.executeLinearLayer(layerIndex, x, self.layersLinear[layerIndex])
			if(debugSmallNetwork):
				print("layerIndex = ", layerIndex)
				print("x after linear = ", x)
			if(layerIndex == self.config.numberOfLayers-1):
				if(not useInbuiltCrossEntropyLossFunction):
					x = self.executeActivationLayer(layerIndex, x, self.layersActivation[layerIndex])	#CHECKTHIS
					x = torch.log(x)
			else:
				x = self.executeActivationLayer(layerIndex, x, self.layersActivation[layerIndex])
			if(debugSmallNetwork):
				print("x after activation = ", x)
		#print("x = ", x)
		#print("y = ", y)
		loss = self.lossFunction(x, y)
		accuracy = self.accuracyFunction(x, y)
		accuracy = accuracy.detach().cpu().numpy()
		
		return loss, accuracy

	def generateLinearLayer(self, layerIndex, config):
		if(layerIndex == 0):
			in_features = config.inputLayerSize
		else:
			if(useLinearSublayers):
				in_features = config.hiddenLayerSize*config.linearSublayersNumber
			else:
				in_features = config.hiddenLayerSize
		if(layerIndex == config.numberOfLayers-1):
			out_features = config.outputLayerSize
		else:
			out_features = config.hiddenLayerSize

		if(self.getUseLinearSublayers(layerIndex)):
			linear = LinearSegregated(in_features=in_features, out_features=out_features, number_sublayers=config.linearSublayersNumber)
		else:
			linear = nn.Linear(in_features=in_features, out_features=out_features)

		self.weightsSetPositiveLayer(layerIndex, linear)

		return linear

	def generateActivationLayer(self, layerIndex, config):
		if(usePositiveWeights):
			if(self.getUseLinearSublayers(layerIndex)):
				activation = nn.Softmax(dim=1)
			else:
				activation = nn.Softmax(dim=1)
		else:
			if(self.getUseLinearSublayers(layerIndex)):
				activation = nn.ReLU()
			else:
				activation = nn.ReLU()		
		return activation

	def executeLinearLayer(self, layerIndex, x, linear):
		self.weightsSetPositiveLayer(layerIndex, linear)	#otherwise need to constrain backprop weight update function to never set weights below 0
		if(self.getUseLinearSublayers(layerIndex)):
			#perform computation for each sublayer independently
			x = x.unsqueeze(dim=1).repeat(1, self.config.linearSublayersNumber, 1)
			x = linear(x)
		else:
			x = linear(x)
		return x

	def executeActivationLayer(self, layerIndex, x, softmax):
		if(self.getUseLinearSublayers(layerIndex)):
			xSublayerList = []
			for sublayerIndex in range(self.config.linearSublayersNumber):
				xSublayer = x[:, sublayerIndex]
				xSublayer = softmax(xSublayer)	#or torch.nn.functional.softmax(xSublayer)
				xSublayerList.append(xSublayer)
			x = torch.concat(xSublayerList, dim=1)
		else:
			x = softmax(x)
		return x
	
	def getUseLinearSublayers(self, layerIndex):
		result = False
		if(useLinearSublayers):
			if(layerIndex != self.config.numberOfLayers-1):	#final layer does not useLinearSublayers
				result = True
		return result

	def weightsSetPositiveLayer(self, layerIndex, linear):
		if(usePositiveWeights):
			if(not usePositiveWeightsClampModel):
				if(self.getUseLinearSublayers(layerIndex)):
					weights = linear.segregatedLinear.weight #only positive weights allowed
					weights = torch.abs(weights)
					linear.segregatedLinear.weight = torch.nn.Parameter(weights)
				else:
					weights = linear.weight #only positive weights allowed
					weights = torch.abs(weights)
					linear.weight = torch.nn.Parameter(weights)
	
	def weightsSetPositiveModel(self):
		if(usePositiveWeights):
			if(usePositiveWeightsClampModel):
				for p in self.parameters():
					p.data.clamp_(0)
			
class LinearSegregated(nn.Module):
	def __init__(self, in_features, out_features, number_sublayers):
		super().__init__()
		self.segregatedLinear = nn.Conv1d(in_channels=in_features*number_sublayers, out_channels=out_features*number_sublayers, kernel_size=1, groups=number_sublayers)
		self.number_sublayers = number_sublayers
		
	def forward(self, x):
		#x.shape = batch_size, number_sublayers, in_features
		x = x.view(x.shape[0], x.shape[1]*x.shape[2], 1)
		x = self.segregatedLinear(x)
		x = x.view(x.shape[0], self.number_sublayers, x.shape[1]//self.number_sublayers)
		#x.shape = batch_size, number_sublayers, out_features
		return x


