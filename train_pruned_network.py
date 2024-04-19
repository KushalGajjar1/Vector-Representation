import torch
import numpy as np
from tqdm import tqdm

from details import device, epochs

def train_pruned_network(network, criterion, optimizer, dataloader):

    network = network.to(device)

    loss_history = []

    for epoch in tqdm(range(epochs)):

        epoch_loss = []

        for img0, img1, label in dataloader:

            img0 = img0.to(device)
            img1 = img1.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output1, output2 = network(img0, img1)
            loss = criterion(output1, output2, label)
            loss.backward()

            optimizer.step()

            epoch_loss.append(loss.item())

        loss_history.append(np.mean(np.array(epoch_loss)))

    return loss_history