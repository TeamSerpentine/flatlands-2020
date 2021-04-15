from abc import abstractmethod

from torch import nn
from torch.types import Device


class Model(nn.Module):
    def __init__(self, device: Device):
        """
        Save the device this model is on
        """
        super(Model, self).__init__()
        self.device = device
        self.to(device)

    @abstractmethod
    def forward(self, x):
        """
        Perform a forward pass on the network
        """
        pass
