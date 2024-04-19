import torch.nn as nn

class Network(nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x1, x2):
        output1 = self.encoder(x1)
        output2 = self.encoder(x2)
        return output1, output2
    
class VectorRepresentation(nn.Module):
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)