import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.input_dim = config.input_dim
        self.hidden_dims = config.hidden_dims
        self.out_dim = config.out_dim
        self.dropout = config.dropout
        self.num_layers = len(self.hidden_dims) + 1
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(self.input_dim, self.hidden_dims[0]), 
                                         nn.ReLU(), 
                                         nn.Dropout(self.dropout)))
        for i in range(1, len(self.hidden_dims)):
            self.layers.append(nn.Sequential(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]),
                                             nn.ReLU(),
                                             nn.Dropout(self.dropout)))
        self.layers.append(nn.Linear(self.hidden_dims[-1], self.out_dim))
        self.nc_only = False
        
    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
        x = self.layers[-1](x)
        return x
    
    
if __name__ == '__main__':
    import easydict
    
    config = easydict.EasyDict({'input_dim': 10, 'hidden_dims': [20, 30], 'out_dim': 2, 'dropout': 0.1})
    model = MLP(config)
    print(model)
    
