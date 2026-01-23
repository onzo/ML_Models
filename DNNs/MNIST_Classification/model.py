import torch.nn as nn
from torch.utils.data import DataLoader

class MNISTClassifier(nn.Module):
    def __init__(self, image_shape=(1,28,28), hidden_inits=(64,32,16)):
        super().__init__()
        input_size = image_shape[0] * image_shape[1] * image_shape[2]
        layers = [nn.Flatten()]
        for hu in hidden_inits:
            layers.append(nn.Linear(input_size, hu))
            layers.append(nn.ReLU())
            input_size = hu
        layers.append(nn.Linear(input_size, 10))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
