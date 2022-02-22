import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvNet(nn.Module):
    def __init__(self, hidden_layers, filters, kernel_size):
        super(ConvNet, self).__init__()
        self.net = []
        self.net.append(nn.Conv2d(1,filters,kernel_size, padding = "same"))
        self.net.append(nn.BatchNorm2d(filters))
        self.net.append(nn.ReLU())
        for i in range(hidden_layers-1):
            self.net.append(nn.Conv2d(filters,filters,kernel_size, padding = "same"))
            self.net.append(nn.BatchNorm2d(filters))
            self.net.append(nn.ReLU())
        
        self.net.append(nn.Conv2d(filters,1,kernel_size, padding = "same"))
        
        self.net = nn.Sequential(*self.net)
    def forward(self, x):
        out = self.net(x)
        return out
    
class ConvDet(nn.Module): 
    def __init__(self,in_channels, matrix_dimension, convolutional_layers, filters, kernel_size, dense_final_layer = False, activation = nn.ReLU()):
        super(ConvDet,self).__init__()
        self.net = [] 
        self.dense_final_layer = dense_final_layer
        #First convolutional input layer 
        self.net.append(nn.Conv2d(in_channels,filters,kernel_size, padding = "same"))
        self.net.append(nn.BatchNorm2d(filters))
        self.net.append(activation)
        
        for i in range(convolutional_layers-1):
            self.net.append(nn.Conv2d(filters,filters,kernel_size, padding = "same"))
            self.net.append(nn.BatchNorm2d(filters))
            self.net.append(activation)
      
        # Last convolutional layer
        self.net.append(nn.Conv2d(filters,1,kernel_size, padding = "same")) 
        
        if self.dense_final_layer: 
            # Add a fully connected layer, potenitially with determinant as bias in the activation
            # Assuming that this means we only have one channel in the input
            self.net.append(nn.Flatten(start_dim = 2))
            self.net.append(DenseLayer(matrix_dimension**2, matrix_dimension**2))
            self.net.append(nn.Unflatten(-1,(matrix_dimension,matrix_dimension)))                       
                     
        
        self.net = nn.Sequential(*self.net)                   
    def forward(self,x): 
        return self.net(x)
                           
class DenseLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, det = 0):
        super().__init__()
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        return F.relu(self.linear(input))

   
                            
        
        