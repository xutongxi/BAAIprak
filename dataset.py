import os
import pickle
import torch
from torch.utils.data import Dataset


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

        return {
            'adj_tensor1': adj_tensor1,
            'attr_tensor1': attr_tensor1,
            'adj_tensor2': adj_tensor2,
            'attr_tensor2': attr_tensor2,
            'label': torch.tensor(label, dtype=torch.float32)
        }


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