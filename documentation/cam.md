# Class activation mappings

Do you know what goes on inside a convolutional network? How does a network decide which one is the car, which one is the cat?

### Introduction
It is difficult to visualize and know the meaning of hidden features in a deep network with respect to the output. CAMs are a great way to check the relation between the image and the output.
The whole learning of neural network relies heavily on the weights. 
CAMs uses the weights as an explanation to the relation between input and label.

### What are CAMs?
> It is a visualization to check what part of the image is relating to label of the image

Class activation mappings (CAMs) are imagerial representation of the weights of the convolutional neural network to visualize the effect of pixels on the output.
It **overlays a heat map over image to describe localization** of the input image to the specific predicted class. 
E.g. The hair, facial hair, eyebrow styles play an important role to classify a person as a male or a female.

#### Examples
![Example of CAMs](https://github.com/jacobgil/keras-grad-cam/raw/master/examples/boat.jpg?raw=true)
![Example of CAMs on MNIST](https://miro.medium.com/max/255/1*o3fkwaqA1l7xKBYnvMru1Q.png)

### Implementation in pytorch

CAMs require a global average pooling layer after the last convolutional layer of the network, followed by a hidden dense layer.
 #### What is a global average pooling layer?
 Support we have a tensor of size 64x64x128. The GAP layer would simple average all the channels to 

Importing the libraries

```
import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn import functional as F
from torch import topk
from matplotlib.pyplot import imshow
import os
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
import skimage.transform
```
To calculate the heat map mask, we need three things:
1. The weights of the final layer
2. The shape of output from the final layer
3. The class predicted for the input 

1. Getting the weights.
```
weight_softmax_params = list(model._modules.get('densenet121').classifier[0].parameters())
weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
```
We get the weights from the layer before activation layer of our model. Store it as a flattened list. 

Note: Converting the tensor variable to .cpu() is important to parse it to other local functions.

We need to save the activation features of the last layer of the network. We attach a hook to save the features of the last layer.
What are activation features? Basically the values to be gotten from network if the network ends at that particular layer.
 ```
 class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()
```
We run one prediction for an image and clip the weights of the trained model.
```
activated_features = SaveFeatures(model._modules.get('densenet121').features.denseblock4.denselayer16)
#while prediction the hook saved the weights
prediction = model(images)
pred_probabilities = F.softmax(prediction).data.squeeze()
activated_features.remove() # removing the hook from the network
```
3. Getting the predicted class index.
The index of the class for the maximum softmax probabilty assigned is stored.

```
# class_idx is the class index, as in to which class has the highest probability in prediction.
class_idx = topk(pred_probabilities,1)[1].int() # Predicting what class each batch belongs too. 0 or 1
```
Now the final doings,
getCAM is the function to dot product the weights for the predicted class, with the feature map and you're all set!

```
def getCAM(feature_conv, weight_fc, class_idx):
    #batch size, no. of channels, height , width
     _, nc, h, w = feature_conv.shape
    #dot product of the weights with the feature map
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    # resize and normalizing
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]
overlay=getCAM(activated_features.features, weight_softmax,class_idx) # clipping the weights and overlaying over the image. 
 ```
 Resizing to the image size and normalizing as the value divided by the max of all values.
 
Finally overlapping the actual image with the heat map. The heat map needs to be normalized back to pixel values [0,255]. 

Images.shape[2:4] is the height and width of the input image. As the shape of the input image categorizes as [batch size, channels, 
height, width] 

Alpha is the opacity, since we are overlapping two images, we keep the opacity of the heat map as 50% .

cmap is the colorization library which is set to 'jet' to get a better visualization in colors.
``` 
img = images[0].cpu().numpy()[0]
imshow(img)
imshow(skimage.transform.resize(overlay[0], images.shape[2:4]), alpha=0.5, cmap='jet');
 ```
### References

[Blog on using CAMs for CNN by Divyanshu Mishra] (https://towardsdatascience.com/demystifying-convolutional-neural-networks-using-class-activation-maps-fe94eda4cef1)

[Pytorch implemenatation of CAMs by Ian Pointer] (http://snappishproductions.com/blog/2018/01/03/class-activation-mapping-in-pytorch.html)
