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
debugSmallNetwork = False

SMANNuseSoftmax = True	#required for useAlgorithmLIANN (disable for performance comparison with standard neural net/relu)
if(SMANNuseSoftmax):
	usePositiveWeights = True	#required
	if(usePositiveWeights):
		usePositiveWeightsClampModel = True	#clamp entire model weights to be positive (rather than per layer); currently required
	useInbuiltCrossEntropyLossFunction = True	#required
		
	simulatedDendriticBranches = True	#optional	#performTopK selection of neurons based on local inhibition - equivalent to multiple independent fully connected weights per neuron (SDBANN)
	if(simulatedDendriticBranches):
		useLinearSublayers = True
		linearSublayersNumber = 1000	#equivalent to numberOfIndependentDendriticBranches
		if(SMANNuseSoftmax):
			normaliseActivationSparsity = True	#increases performance
		else:
			normaliseActivationSparsity = True	#required
			
		trainLastLayerOnly = False	#optional	#LUANN
		
	useLUANNonly = False
else:
	#debug only (emulate LUANNpt_LUANN codebase);
	useLUANNonly = True
	if(useLUANNonly):
		useLinearSublayers = True
		linearSublayersNumber = 1000	#equivalent to numberOfIndependentDendriticBranches

		trainLastLayerOnly = True	#optional	#LUANN

if(trainLastLayerOnly):
	batchSize = 64
	numberOfLayers = 4
	hiddenLayerSize = 1000

	normaliseActivationSparsity = True	#increases performance


workingDrive = '/large/source/ANNpython/LIANNpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelLIANN'

