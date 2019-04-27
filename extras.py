from PIL import Image
from matplotlib.pyplot import imshow
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch import topk
import numpy as np
import skimage.transform
import numpy as np
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

def get_tensor(image_name,resize=(224,224)):
  my_transforms=transforms.Compose([
      transforms.RandomResizedCrop(max((resize))),
      transforms.ToTensor(),
      transforms.Normalize([0.485,0.486,0.406],[0.229,0.224,0.225])
  ])
  image = Image.open(image_name)
  return(my_transforms(image).unsqueeze(0))

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()

def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]
