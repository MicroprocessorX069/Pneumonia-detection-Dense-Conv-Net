from flask import Flask,render_template,flash ,redirect,url_for, session, logging,request
#from flask_mysqldb import MySQL
#from wtforms import Form, StringField, TextAreaField, PasswordField, validators,SubmitField, Field
#from wtforms import TextField
#from passlib.hash import sha256_crypt
from functools import wraps
from inference import get_tensor
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
from matplotlib.pyplot import imshow,savefig
from torch import topk
from torch.nn import functional as F
import skimage.transform
import scipy.misc
from model import DenseNet121
from extras import getCAM, SaveFeatures
from inference import get_tensor
import inference

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
	if request.method=='GET':

		return render_template('temp.html',value="Something")
	if request.method=='POST':
		if 'file' not in request.files:
			print('file not uploaded')
			return
		file = request.files['file']
		image = file.read()
		device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		images=get_tensor(image)
		images=images.to(device)
		img = images[0].cpu().numpy()[0]

		class_idx,cam=get_prediction(image)
		new_img = Image.blend(scipy.misc.toimage(img), scipy.misc.toimage(cam), 0.5)
		imshow(img,cmap='gray')
		imshow(cam,alpha=0.2,cmap='jet')
		savefig("static/outpit3.png")
		return render_template('index.html',value=str(class_idx))

def get_prediction(image_name="pn.png"):
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_classes=2
    images=get_tensor(image_name)
    images=images.to(device)
    model = DenseNet121(num_classes).to(device)
    model.load_state_dict(torch.load("pneumonia_model.pth",map_location="cpu"))

    prediction=model(images.to(device))
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
    print(img.shape)
    #imshow(img,cmap='gray')
    #imshow(skimage.transform.resize(overlay[0], images.shape[2:4]), alpha=0.4, cmap='jet')
    return class_idx,skimage.transform.resize(overlay[0], images.shape[2:4])


if __name__=="__main__":
    app.secret_key='secret123'
    app.run(debug=True)
