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

#debug only (emulate LUANNpt_LUANN codebase);
useLUANNonly = False
LIANNlocalLearning = False
if(useLUANNonly):
	activationFunctionType = "relu"
	useLinearSublayers = True
	linearSublayersNumber = 1000	#equivalent to numberOfIndependentDendriticBranches

	trainLastLayerOnly = True	#optional	#LUANN
else:
	activationFunctionType = "softmax"	#required for useAlgorithmLIANN (disable for performance comparison with standard neural net/relu)
	#activationFunctionType = "none"
	usePositiveWeights = True	#required
	if(usePositiveWeights):
		usePositiveWeightsClampModel = True	#clamp entire model weights to be positive (rather than per layer); currently required
	useInbuiltCrossEntropyLossFunction = True	#required
		
	simulatedDendriticBranches = True	#optional	#performTopK selection of neurons based on local inhibition - equivalent to multiple independent fully connected weights per neuron (SDBANN)
	if(simulatedDendriticBranches):
		useLinearSublayers = True
		linearSublayersNumber = 1000	#equivalent to numberOfIndependentDendriticBranches
		if(activationFunctionType=="softmax"):
			normaliseActivationSparsity = True	#increases performance
		else:
			normaliseActivationSparsity = True	#required
			
		trainLastLayerOnly = True	#optional	#LUANN
		if(trainLastLayerOnly):
			LIANNlocalLearning = True
			#normaliseActivationSparsity = False
			if(LIANNlocalLearning):
				LIANNlocalLearningNeuronActiveThreshold = 0.1	#minimum activation level for neuron to be considered active (required for usePositiveWeights net)	#CHECKTHIS
				LIANNlocalLearningRate = 0.001	#0.01	#default: 0.001	#CHECKTHIS
				LIANNlocalLearningBias = False	#bias learning towards most signficant weights
		

if(trainLastLayerOnly):
	batchSize = 64
	numberOfLayers = 4
	hiddenLayerSize = 1000

	normaliseActivationSparsity = True	#increases performance


workingDrive = '/large/source/ANNpython/LIANNpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelLIANN'

