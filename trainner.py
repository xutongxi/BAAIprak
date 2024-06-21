import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import os

import model
import lossfunction


class Trainer():
    def __init__(self, GEN: model.GraphEmbeddingNetwork, GENdataLoader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), with_cuda=True, cuda_devices=None, log_freq: int=10):
        # test if cuda could be used
        cuda_condition = torch.cuda.is_available() and with_cuda
        print(f'CUDA available: {cuda_condition}')
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        # upload model to device
        self.model = GEN.to(self.device)
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # store dataset
        self.train_data = GENdataLoader
        self.test_data = test_dataloader

        self.optimizer = Adam(self.model.parameters(), lr=lr, betas=betas)
        self.criterion = lossfunction.CosSimLoss()
        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

# epoch指的是目前进行到了第几论，而不是一共要进行第几论
    def train(self, epoch, batch_size=10):
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(self.train_data),
                              desc="EP_%s:%d" % ("train", epoch),
                              total=len(self.train_data),
                              bar_format="{l_bar}{r_bar}")
        for i, data in data_iter:
            if isinstance(data, dict):
                data = {key: value.to(self.device) for key, value in data.items()}
            else:
                print(f"Unexpected data format at index {i}: {data}")
                continue
            # data = {key: value.to(self.device) for key, value in data.items()}
            print(type(data.values()))
            tensor_u1 = torch.zeros(batch_size, data["adj_tensor1"].size(1), self.model.embedding_size).to(self.device)
            tensor_u2 = torch.zeros(batch_size, data["adj_tensor2"].size(1), self.model.embedding_size).to(self.device)
            # print(tensor_u1.shape)
            # print(tensor_u2.shape)
            attribute_vector1 = self.model.forward(data["attr_tensor1"], data["adj_tensor1"], tensor_u1)
            attribute_vector2 = self.model.forward(data["attr_tensor2"], data["adj_tensor2"], tensor_u2)
            loss = self.criterion.forward(attribute_vector1, attribute_vector2, data["label"])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
