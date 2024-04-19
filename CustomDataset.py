import torch
from torch.utils.data import Dataset
import numpy as np

class CustomDatasetMNIST(Dataset):

    def __init__(self, mnist_dataset, transform=None):
        self.mnist_dataset = mnist_dataset
        self.transform = transform

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, index):
        img0, label0 = self.mnist_dataset[index]
        
        same_class = np.random.randint(2)
        
        if same_class:
            while True:
                img1, label1 = self.mnist_dataset[np.random.randint(0, len(self.mnist_dataset)-1)]
                if label0 == label1:
                    break

        else:
            while True:
                img1, label1 = self.mnist_dataset[np.random.randint(0, len(self.mnist_dataset)-1)]
                if label0 != label1:
                    break
        
        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.tensor(int(label0 != label1), dtype=torch.float32)
    

class CustomDatasetOffice(Dataset):

    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return self.label.shape[0]
    
    def __getitem__(self, index):
        img0 = self.data[index]
        label0 = self.label[index]

        same_class = np.random.randint(2)

        if same_class:
            while True:
                idx = np.random.randint(0, self.data.shape[0]-1)
                img1 = self.data[idx]
                label1 = self.label[idx]
                if label0 == label1:
                    break

        else:
            while True:
                idx = np.random.randint(0, self.data.shape[0]-1)
                img1 = self.data[idx]
                label1 = self.label[idx]
                if label0 != label1:
                    break

        # if self.transform:
        #     img0 = self.transform(img0)
        #     img1 = self.transform(img1)

        # return img0, img1, torch.from_numpy(np.array([int(label0 != label1)], dtype=np.float32))
        return torch.tensor(img0), torch.tensor(img1), torch.from_numpy(np.array([int(label0 != label1)], dtype=np.float32))