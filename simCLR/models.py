import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNetSimCLR(nn.Module):
    def __init__(self, out_dim = 128):
        super().__init__()
        base = models.resnet18(weights=None) # training from scratch
        dim_mlp = base.fc.in_features # base.fc = Linear(512, 1000) in ResNet18 , so it gives the size of the feature vector entering that classifier.
        base.fc = nn.Identity() # remove the final classifier layer. Now the ResNet outputs a feature vector of size 512 insted of a logit for 1000 classes.

        self.encoder = base
        self.projection_head = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), #first linear layer that maps the feature vector to a higher dimensional space
            nn.ReLU(), #nonlinear activation function
            nn.Linear(dim_mlp, out_dim) #final linear layer that maps the feature vector to the output dimension
        )
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        #normalize the output of the projection head to make it a unit vector
        z = F.normalize(z, dim=1)
        return h,z
