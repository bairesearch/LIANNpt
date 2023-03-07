"""LIANNpt_SMANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see LIANNpt_main.py

# Usage:
see LIANNpt_main.py

# Description:
LIANNpt_SMANN softmax artificial neural network (SMANN) model

"""

from LIANNpt_globalDefs import *
import LIANNpt_SMANNmodel
import LIANNpt_data

def createModel(dataset):
	datasetSize = LIANNpt_data.getDatasetSize(dataset, printSize=True)
	numberOfFeatures = LIANNpt_data.countNumberFeatures(dataset)
	numberOfClasses, numberOfClassSamples = LIANNpt_data.countNumberClasses(dataset)
	
	print("creating new model")
	config = LIANNpt_SMANNmodel.SMANNconfig(
		batchSize = batchSize,
		numberOfLayers = numberOfLayers,
		hiddenLayerSize = hiddenLayerSize,
		inputLayerSize = numberOfFeatures,
		outputLayerSize = numberOfClasses,
		linearSublayersNumber = linearSublayersNumber,
		numberOfFeatures = numberOfFeatures,
		numberOfClasses = numberOfClasses,
		datasetSize = datasetSize,
		numberOfClassSamples = numberOfClassSamples
	)
	model = LIANNpt_SMANNmodel.SMANNmodel(config)
	return model
	
