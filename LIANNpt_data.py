"""LIANNpt_data.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see LIANNpt_main.py

# Usage:
see LIANNpt_main.py

# Description:
LIANNpt data 

"""


import torch
from datasets import load_dataset
from LIANNpt_globalDefs import *

def loadDataset():
	base_url = "https://huggingface.co/datasets/wwydmanski/blog-feedback/resolve/main/"	
	dataset = load_dataset('csv', data_files={"train": base_url + "train.csv", "test": base_url + "test.csv"})
	#print("dataset['train'] = ", dataset['train'])
	return dataset

def countNumberClasses(dataset):
	datasetSize = getDatasetSize(dataset)
	numberOfClasses = 0
	for i in range(datasetSize):
		row = dataset[i]
		target = int(row['target'])
		#print("target = ", target)
		if(target > numberOfClasses):
			numberOfClasses = target
	print("numberOfClasses = ", numberOfClasses)
	return numberOfClasses

def countNumberFeatures(dataset):
	numberOfFeatures = len(dataset.features)-1	#-1 to ignore class targets
	print("numberOfFeatures = ", numberOfFeatures)
	return numberOfFeatures
	
def getDatasetSize(dataset):
	datasetSize = dataset.num_rows
	return datasetSize
	
def createDataLoader(dataset):
	dataLoaderDataset = DataloaderDatasetInternet(dataset)	
	loader = torch.utils.data.DataLoader(dataLoaderDataset, batch_size=batchSize, shuffle=True)	#shuffle not supported by DataloaderDatasetHDD

	#loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True)
	return loader

class DataloaderDatasetInternet(torch.utils.data.Dataset):
	def __init__(self, dataset):
		self.datasetSize = getDatasetSize(dataset)
		self.datasetIterator = iter(dataset)
			
	def __len__(self):
		return self.datasetSize

	def __getitem__(self, i):
		document = next(self.datasetIterator)
		documentList = list(document.values())
		x = documentList[0:-1]
		y = documentList[-1]
		x = torch.Tensor(x).float()
		batchSample = (x, y)
		return batchSample
		
