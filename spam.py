#
# device = next(self.parameters()).device
# u = {node: torch.zeros(self.embedding_size, device=device) for node in x.nodes}
# l_v = {node: torch.zeros(self.embedding_size, device=device) for node in x.nodes}
# summary = torch.zeros(self.embedding_size, device=device)
#
# for _ in range(spreads):
#     for node in x.nodes:
#         l_v[node] = torch.zeros(self.embedding_size)
#         for predecessor in x.predecessors(node):
#             l_v[node] += u[predecessor]
#
#     for node in x.nodes:
#         for linear in self.linear_blockP:  # additional relu are added
#             l_v[node] = F.relu(linear(l_v[node]))
#         l_v[node] = self.linear_add_p(l_v[node])
#         u[node] = self.linearW1(x[node]) + l_v[node]
# for node in x.nodes:
#     summary += u[node]
# return self.linearW2(summary)
