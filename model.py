
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from PIL import Image
import io
import sys
from matplotlib.pyplot import imshow
from torch import topk
from torch.nn import functional as F

class DenseNet121(nn.Module):
  def __init__(self,out_size=2):
    super(DenseNet121,self).__init__()
    self.densenet121=torchvision.models.densenet121(pretrained=True)
    num_features=self.densenet121.classifier.in_features
    self.densenet121.classifier=nn.Sequential(
    nn.Linear(num_features,out_size),
    nn.Sigmoid())
  def forward(self,x):
    x=self.densenet121(x)
    return x
