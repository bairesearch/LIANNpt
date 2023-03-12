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

SMANNuseSoftmax = True	#required for useAlgorithmSMANN (disable for performance comparison with standard neural net/relu)
if(SMANNuseSoftmax):
	usePositiveWeights = True	#required
	if(usePositiveWeights):
		usePositiveWeightsClampModel = True	#clamp entire model weights to be positive (rather than per layer); currently required
	useInbuiltCrossEntropyLossFunction = True	#optional
else:
	usePositiveWeights = False	#required
	useInbuiltCrossEntropyLossFunction = True	#required

workingDrive = '/large/source/ANNpython/LIANNpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelLIANN'
