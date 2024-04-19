import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from details import epochs, device, batch_size
from Network import VectorRepresentation
from Encoder import Encoder
from Classifier import Classifier
from data_classifier import train_loader

encoder = Encoder()
vectorModel = VectorRepresentation(encoder)
vectorModel.load_state_dict(torch.load('network.pt'))

classifier = Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

classifier = classifier.to(device)
vectorModel = vectorModel.to(device)

train_acc = []

for epoch in tqdm(range(epochs)):

    train_epoch_acc = []

    for input, label in train_loader:

        input = input.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        outputs = classifier(vectorModel(input))
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_hat = torch.argmax(outputs, 1)
            score = torch.eq(y_hat, label).sum()
            train_epoch_acc.append((score.item()/batch_size)*100)
    
    with torch.no_grad():
        train_acc.append(np.mean(np.array(train_epoch_acc)))

torch.save(classifier.state_dict(), 'classifier.pt')

plt.plot(train_acc)
plt.show()