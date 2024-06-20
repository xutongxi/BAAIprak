import os
import pickle
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class GraphDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        file_path = os.path.join(self.data_dir, data_file)

        with open(file_path, 'rb') as f:
            adj_tensor1, attr_tensor1, adj_tensor2, attr_tensor2, label = pickle.load(f)
        # print(type(adj_tensor1), type(adj_tensor2), type(attr_tensor1), type(attr_tensor2), type(label))
        # print(adj_tensor1.shape, attr_tensor1.shape, adj_tensor2.shape, attr_tensor2.shape, label.shape)
        return {
            'adj_tensor1': adj_tensor1,
            'attr_tensor1': attr_tensor1,
            'adj_tensor2': adj_tensor2,
            'attr_tensor2': attr_tensor2,
            'label': torch.tensor(label, dtype=torch.float32)
        }


def custom_collate_fn(batch):
    def pad_tensor(tensor, target_shape):
        padding = [0] * 4
        padding[1] = target_shape[1] - tensor.shape[1]
        padding[3] = target_shape[0] - tensor.shape[0]
        # print(padding, tensor.shape)
        resul = F.pad(tensor, padding, "constant", 0)
        # print(target_shape, resul.shape)
        return resul

    # 找到每个张量的最大形状
    max_shape = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor) and batch[0][key].dim() > 0:
            max_shape[key] = [max(item[key].shape[i] for item in batch) for i in range(len(batch[0][key].shape))]
            print(key, batch[0][key].shape)
        else:
            print(key, batch[0][key].shape)

    # 对每个样本的每个张量进行填充，并合并到一个批次中
    collated_batch = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor) and batch[0][key].dim() > 0:
            padded_tensors = [pad_tensor(item[key], max_shape[key]) for item in batch]
            collated_batch[key] = torch.stack(padded_tensors, dim=0)
        else:
            collated_batch[key] = torch.stack([item[key] for item in batch])
    # for key in batch[0].keys():
    #     if key is 'label':
    #         max_shape[key] = len(batch[0][key])
    #     else:
    #         max_shape[key] = [max(item[key].shape[i] for item in batch) for i in range(len(batch[0][key].shape))]
    #
    # # 对每个样本的每个张量进行填充，并合并到一个批次中
    # collated_batch = {}
    # for key in batch[0].keys():
    #     if 'label' in key:
    #         collated_batch[key] = torch.stack([item[key] for item in batch])
    #     else:
    #         padded_tensors = [pad_tensor(item[key], max_shape[key]) for item in batch]
    #         collated_batch[key] = torch.stack(padded_tensors, dim=0)
    assert batch[0]['adj_tensor1'].size(0) == batch[0]['attr_tensor1'].size(0)
    return collated_batch

# 示例用法
if __name__ == "__main__":
    data_dir = r"C:\Users\93293\OneDrive\桌面\FILE\LMU课程\SS24\Prak PL&AI\BAAIprak-main\BAAIprak-main\dataset"
    dataset = GraphDataset(data_dir)

    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]

    print("Sample keys:", sample.keys())
    print("Adjacency Tensor 1 shape:", sample['adj_tensor1'].shape)
    print("Attribute Tensor 1 shape:", sample['attr_tensor1'].shape)
    print("Adjacency Tensor 2 shape:", sample['adj_tensor2'].shape)
    print("Attribute Tensor 2 shape:", sample['attr_tensor2'].shape)
    print("Label:", sample['label'])