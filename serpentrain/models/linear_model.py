import torch.nn.functional as F
from torch import nn, tensor

from serpentrain.models.model import Model


class LinearModel(Model):
    """
    Neural network with fully connected hidden layers with relu on all hidden layers (but none on the output layer)
    """

    def __init__(self, device, input_size, layer_sizes, output_size):
        super(LinearModel, self).__init__(device)

        self.input_size = input_size
        self.output_size = output_size
        self.layer_sizes = layer_sizes

        self.layers = nn.ModuleList()

        # Construct hidden layers
        prev_layer_size = input_size
        for layer_size in layer_sizes:
            self.layers.append(nn.Linear(prev_layer_size, layer_size))
            prev_layer_size = layer_size

        # Construct output layer
        self.layers.append(nn.Linear(prev_layer_size, output_size))

        self.debug = False

        # Start the model in evaluation mode
        self.eval()
        self.to(self.device)

    def forward(self, state: tensor):
        # Perform relu on all but the last layer
        layer_output = state
        for i in range(0, len(self.layers) - 1):
            layer = self.layers[i]
            layer_output = F.relu(layer(layer_output))
        output = self.layers[-1](layer_output)
        return output
