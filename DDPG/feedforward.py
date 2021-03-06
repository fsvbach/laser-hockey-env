import torch
import numpy as np

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, actor):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes  = hidden_sizes
        self.output_size  = output_size
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList(
            [ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])] + [torch.nn.Linear(self.hidden_sizes[-1], self.output_size)])
        self.actor = actor
        if self.actor:
            self.activations = [ torch.nn.ReLU() for l in  self.layers ] + [torch.nn.Tanh()]
            torch.nn.init.uniform_(self.layers[-1].weight, -0.003, 0.003)
        else:
            self.activations = [ torch.nn.LeakyReLU() for l in  self.layers ]
            torch.nn.init.kaiming_uniform_(self.layers[-1].weight, a=1e-3, mode='fan_out', nonlinearity='leaky_relu')

        torch.nn.init.uniform_(self.layers[-1].bias, -0.0003, 0.0003)
 

    def forward(self, x):
        for layer,activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        return x



    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()
