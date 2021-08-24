import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, alpha):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, anchor, positive, negative, size_average=True):      
        
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_reg = cos(negative - anchor, positive - anchor).sum(0) 
        
        #losses = F.relu(distance_positive - distance_negative + self.margin - self.alpha * cos_reg)
        losses = F.softplus(distance_positive - distance_negative + self.margin - self.alpha * cos_reg)

        return losses.mean()
    

def pairwise_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)
  
