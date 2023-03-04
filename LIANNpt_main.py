"""LIANNpt_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
conda create -n pytorchsenv
source activate pytorchsenv
conda install python
pip install datasets
pip install torch
pip install lovely-tensors
pip install torchmetrics

# Usage:
source activate pytorchsenv
python LIANNpt_main.py

# Description:
LIANNpt main - Local Inhibitory artificial neural network (softmax network)

"""

import torch
from tqdm.auto import tqdm
from torch import optim

from LIANNpt_globalDefs import *
import LIANNpt_SMANN
import LIANNpt_data

#https://huggingface.co/docs/datasets/tabular_load

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main():
	dataset = LIANNpt_data.loadDataset()
	if(stateTrainDataset):
		model = LIANNpt_SMANN.createModel(dataset['train'])	#dataset['test'] not possible as test does not contain all classes
		processDataset(True, dataset['train'], model)
	if(stateTestDataset):
		model = loadModel()
		processDataset(False, dataset['test'], model)

def processDataset(trainOrTest, dataset, model):

	if(trainOrTest):
		optim = torch.optim.Adam(model.parameters(), lr=learningRate)
		model.to(device)
		model.train()
		numberOfEpochs = trainNumberOfEpochs
	else:
		model.to(device)
		model.eval()
		numberOfEpochs = 1
		
	for epoch in range(numberOfEpochs):
		loader = LIANNpt_data.createDataLoader(dataset)	#required to reset dataloader and still support tqdm modification
		loop = tqdm(loader, leave=True)
		
		if(printAccuracyRunningAverage):
			(runningLoss, runningAccuracy) = (0.0, 0.0)
		
		for batchIndex, batch in enumerate(loop):
			if(trainOrTest):
				loss, accuracy = trainBatch(batchIndex, batch, model, optim)
			else:
				loss, accuracy = testBatch(batchIndex, batch, model)
			
			if(printAccuracyRunningAverage):
				(loss, accuracy) = (runningLoss, runningAccuracy) = (runningLoss/runningAverageBatches*(runningAverageBatches-1)+(loss/runningAverageBatches), runningAccuracy/runningAverageBatches*(runningAverageBatches-1)+(accuracy/runningAverageBatches))
				
			loop.set_description(f'Epoch {epoch}')
			loop.set_postfix(batchIndex=batchIndex, loss=loss, accuracy=accuracy)

		saveModel(model)
					
def trainBatch(batchIndex, batch, model, optim):

	optim.zero_grad()

	loss, accuracy = propagate(batchIndex, batch, model)
	loss.backward()
	optim.step()
	
	if(SMANNusePositiveWeights):
		if(SMANNusePositiveWeightsClampModel):
			model.weightsSetPositiveModel()

	if(batchIndex % modelSaveNumberOfBatches == 0):
		saveModel(model)
	loss = loss.item()
			
	return loss, accuracy
			
def testBatch(batchIndex, batch, model):

	loss, accuracy = propagate(batchIndex, batch, model)

	loss = loss.detach().cpu().numpy()
	
	return loss, accuracy

def saveModel(model):
	torch.save(model, modelPathNameFull)

def loadModel():
	print("loading existing model")
	model = torch.load(modelPathNameFull)
	return model
		
def propagate(batchIndex, batch, model):
	(x, y) = batch
	y = y.long()
	x = x.to(device)
	y = y.to(device)
	#print("x = ", x)
	#print("y = ", y)
	loss, accuracy = model(x, y)
	return loss, accuracy
				
if(__name__ == '__main__'):
	main()






