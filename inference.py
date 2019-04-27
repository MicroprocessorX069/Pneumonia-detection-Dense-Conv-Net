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
from model import DenseNet121
from extras import getCAM, SaveFeatures

def get_tensor(image_bytes,resize=(224,224)):
  my_transforms=transforms.Compose([
      transforms.RandomResizedCrop(max((resize))),
      transforms.ToTensor(),
      transforms.Normalize([0.485,0.486,0.406],[0.229,0.224,0.225])
  ])
  image = Image.open(io.BytesIO(image_bytes))
  return(my_transforms(image).unsqueeze(0))


def get_prediction(image_bytes):
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_classes=2
    images=get_tensor(image_bytes)
    images=images.to(device)
    model = DenseNet121(num_classes).to(device)
    model.load_state_dict(torch.load("/content/pneumonia_model.pth"))

    prediction=model(something.to(device))
    activated_features = SaveFeatures(model._modules.get('densenet121').features.denseblock4.denselayer16)
    prediction = model(images)
    pred_probabilities = F.softmax(prediction).data.squeeze()
    activated_features.remove()
    topk(pred_probabilities,1)

    weight_softmax_params = list(model._modules.get('densenet121').classifier[0].parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
    #weight_softmax*=0.01
    class_idx = topk(pred_probabilities,1)[1].int()
    overlay=getCAM(activated_features.features, weight_softmax,class_idx)

    img = images[0].cpu().numpy()[0]
    #imshow(img,cmap='gray')
    #imshow(skimage.transform.resize(overlay[0], images.shape[2:4]), alpha=0.4, cmap='jet')
    return class_idx,skimage.transform.resize(overlay[0], images.shape[2:4])
