import torch
import torch.nn as nn
import networkx as nx
import torch.nn.functional as F


class DepthEmbeddingNetwork(nn.Module):
    def __init__(self, embedded_depth, vector_size, embedding_size):
        super(DepthEmbeddingNetwork, self).__init__()
        self.embedded_depth = embedded_depth
        # self.spreads = spreads
        self.embedding_size = embedding_size
        self.linearW1 = nn.Linear(vector_size, embedding_size)
        self.linear_blockP = nn.ModuleList([nn.Linear(embedding_size, embedding_size) for _ in range(embedded_depth - 1)])
        self.linear_add_p = nn.Linear(embedding_size, embedding_size)
        self._initialize_weights()

    def forward(self, attr_tensor, adj_tensor, tensor_u):
        # adj_tensor = torch.transpose(attr_tensor,0, 1)
        print(adj_tensor.shape, tensor_u.shape)
        tensor_u = torch.matmul(adj_tensor, tensor_u)
        for layer in self.linear_blockP:
            tensor_u = F.relu(layer(tensor_u))
        tensor_u = self.linear_add_p(tensor_u)
        influence_x = self.linearW1(attr_tensor)
        tensor_u = F.tanh(influence_x + tensor_u)
        return tensor_u

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in self.linear_blockP:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
class GraphEmbeddingNetwork(nn.Module):
    def __init__(self, spread_times, embedded_depth, vector_size, embedding_size):
        super(GraphEmbeddingNetwork, self).__init__()
        self.spread_times = spread_times
        self.embedding_size = embedding_size
        self.spreads_network = DepthEmbeddingNetwork(embedded_depth, vector_size, embedding_size)
        self.linearW2 = nn.Linear(embedding_size, embedding_size)
        nn.init.kaiming_normal_(self.linearW2.weight, nonlinearity='relu')

    def forward(self, attr_tensor, adj_tensor, tensor_u):
        for times in range(self.spread_times):
            tensor_u = self.spreads_network(attr_tensor, adj_tensor, tensor_u)
        sum_tensor = torch.sum(tensor_u, dim=1)
        # print(tensor_u.shape, sum_tensor.shape)
        # print("看一下形状")
        return self.linearW2(sum_tensor)

