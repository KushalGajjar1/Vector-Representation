import torch
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from details import batch_size

data_source = np.load('data/source_data.npy')
label_source = np.load('data/source_label.npy')

data_target = np.load('data/target_data.npy')
label_target = np.load('data/target_label.npy')

data_source_norm = data_source / np.max(data_source)
data_target_norm = data_target / np.max(data_target)

data_source_tensor_initial = torch.tensor(data_source_norm).float()
data_source_tensor = torch.transpose(data_source_tensor_initial, 3, 1)

label_source_tensor = torch.tensor(label_source).long()

data_target_tensor_initial = torch.tensor(data_target_norm).float()
data_target_tensor = torch.transpose(data_target_tensor_initial, 3, 1)

label_target_tensor = torch.tensor(label_target).long()

train_data = data_source_tensor
train_label = label_source_tensor

test_data = data_target_tensor
test_label = label_target_tensor

train_data = TensorDataset(train_data,train_label)
test_data = TensorDataset(test_data,test_label)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size)