import torch.nn as nn 

class Projection(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=768, output_dim=768):
        super().__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.model = nn.Sequential(
            # nn.Linear(self.input_dim, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        )
    
    def forward(self, x):
        x = self.model(x)
        return nn.functional.normalize(x, dim=1)