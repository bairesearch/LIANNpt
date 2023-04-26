"""LIANNpt_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
LIANNpt globalDefs

"""

debugUsePositiveWeightsVerify = False

trainLastLayerOnly = False	#initialise (dependent var)

SMANNuseSoftmax = True	#required for useAlgorithmLIANN (disable for performance comparison with standard neural net/relu)
if(SMANNuseSoftmax):
	usePositiveWeights = True	#required
	if(usePositiveWeights):
		usePositiveWeightsClampModel = True	#clamp entire model weights to be positive (rather than per layer); currently required
	useInbuiltCrossEntropyLossFunction = True	#required
		
	simulatedDendriticBranches = False	#optional	#performTopK selection of neurons based on local inhibition - equivalent to multiple independent fully connected weights per neuron (SDBANN)
	if(simulatedDendriticBranches):
		useLinearSublayers = True
		linearSublayersNumber = 100	#equivalent to numberOfIndependentDendriticBranches
		normaliseActivationSparsity = False	#initialise (dependent var)
		if(not SMANNuseSoftmax):
			normaliseActivationSparsity = True

		trainLastLayerOnly = False	#optional	#LUANN

workingDrive = '/large/source/ANNpython/LIANNpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelLIANN'

