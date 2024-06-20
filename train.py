import model
import dataset
import trainner

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

spread_times = 3
embedded_depth = 2  # at least 1
vector_size = 7
embedding_size = 64

train_dataset = "data/training/matrix_pairs"
test_dataset = None
output_path = "parameter/"

print("Creating Dataset")
train_dataset = dataset.GraphDataset(train_dataset)
test_dataset = dataset.GraphDataset(test_dataset) if test_dataset is not None else None

print("Creating Dataloader")
train_data_loader = DataLoader(train_dataset, batch_size=10, num_workers=10)
test_data_loader = DataLoader(test_dataset, batch_size=10, num_workers=10) \
    if test_dataset is not None else None
print("Building model")
model = model.GraphEmbeddingNetwork(spread_times, embedded_depth, vector_size, embedding_size)

print("Creating Trainer")
train_model = trainner.Trainer(model, train_data_loader, test_data_loader)

print("Training Start")
for epoch in range(20):
    train_model.train(epoch)
    train_model.save(epoch, output_path)