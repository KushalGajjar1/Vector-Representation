import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(512, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        layer1 = F.relu(self.input(x))
        layer2 = F.relu(self.fc1(layer1))
        layer3 = F.relu(self.fc2(layer2))
        layer4 = F.relu(self.fc3(layer3))
        layer5 = self.output(layer4)
        return layer5