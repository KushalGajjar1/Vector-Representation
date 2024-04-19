import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(256*14*14, 1024)
        self.fc2 = nn.Linear(1024, 512)

    def forward(self, x):

        layer1 = F.relu(self.conv1(x))
        layer2 = F.relu(self.conv2(layer1))
        layer3 = F.relu(self.conv3(layer2))
        layer4 = self.conv4(layer3)
        layer4_p = layer4.view(-1, int(layer4.nelement()/layer4.shape[0]))
        layer5 = F.relu(self.fc1(layer4_p))
        layer6 = self.fc2(layer5)
        return layer6