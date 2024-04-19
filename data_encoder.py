import torch
import numpy as np
from torch.utils.data import DataLoader

from CustomDataset import CustomDatasetOffice
from details import batch_size

data_source = np.load('data/amazon_data.npy')
label_source = np.load('data/amazon_label.npy')

data_source_norm = data_source / np.max(data_source)

data_source_tensor_initial = torch.tensor(data_source_norm).float()
data_source_tensor = torch.transpose(data_source_tensor_initial, 3, 1)

label_source_tensor = torch.tensor(label_source).long()

train_data = data_source_tensor
train_label = label_source_tensor

train_data = CustomDatasetOffice(torch.Tensor.numpy(train_data), torch.Tensor.numpy(train_label))
dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
