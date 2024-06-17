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

    def forward(self, attr_tensor, adj_tensor, tensor_u):
        # adj_tensor = torch.transpose(attr_tensor,0, 1)
        tensor_u = torch.matmul(adj_tensor, tensor_u)
        for layer in self.linear_blockP:
            tensor_u = F.relu(layer(tensor_u))
        tensor_u = self.linear_add_p(tensor_u)
        influence_x = self.linearW1(attr_tensor)
        tensor_u = F.tanh(influence_x + tensor_u)
        return tensor_u


class GraphEmbeddingNetwork(nn.Module):
    def __init__(self, spread_times, embedded_depth, vector_size, embedding_size):
        super(GraphEmbeddingNetwork, self).__init__()
        self.spread_times = spread_times
        self.spreads_network = DepthEmbeddingNetwork(embedded_depth, vector_size, embedding_size)
        self.linearW2 = nn.Linear(embedding_size, embedding_size)

    def forward(self, attr_tensor, adj_tensor, tensor_u):
        for times in range(self.spread_times):
            tensor_u = self.spreads_network(attr_tensor, adj_tensor, tensor_u)
        sum_tensor = torch.sum(tensor_u, dim=0)
        return self.linearW2(sum_tensor)

