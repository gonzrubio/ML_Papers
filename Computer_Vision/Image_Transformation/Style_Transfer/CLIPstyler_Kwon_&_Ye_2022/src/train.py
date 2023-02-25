#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:24:31 2023

@author: gonzalo
"""

# def preprocess(image_path):
#     image = Image.open(image_path)
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )
#     ])
#     image = transform(image).unsqueeze(0)
#     return image
# 
# content_image = preprocess("path/to/content/image.jpg")


def get_features(image, model, layers=None):

    if layers is None:
        layers = {'0': 'conv1_1',  
                  '5': 'conv2_1',  
                  '10': 'conv3_1', 
                  '19': 'conv4_1', 
                  '21': 'conv4_2', 
                  '28': 'conv5_1',
                  '31': 'conv5_2'
                 }  
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)   
        if name in layers:
            features[layers[name]] = x
    
    return features

for name, module in model.named_modules():
    print(name, type(module))
# TODO vgg from torchvision
# TODO ask chatgpt if I can use another SOTA convnet or transformer

from torchvision.models import vgg19_bn, VGG19_BN_Weights
import torch.nn.functional as F
with torch.no_grad():
    weights = VGG19_BN_Weights.DEFAULT
    preprocess = weights.transforms()
    model = vgg19_bn(weights=weights)
    model.eval()
    img_transformed = preprocess(img)
    content_image = preprocess("path/to/content/image.jpg")
    features = model(content_image)

    features4_2 = model.features[:23](content_image)
    features5_2 = model.features[:30](content_image)
    content_loss = F.mse_loss(features4_2, features4_2) + F.mse_loss(features5_2, features5_2)

# TODO clip from transformers or opeinai
# TODO see how authors use pre-trained models
# TODO set up W&B

# from torchvision import models as M
# from torchvision.transforms import transforms as T

# from PIL import Image
# from torchvision models import such and such


# preprocess = T.Compose([
#     T.Resize([256, ]),
#     T.CenterCrop(224),
#     T.PILToTensor(),
#     T.ConvertImageDtype(torch.float),
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])


# data_loader = DataLoader(ImageFolder(dataset_path, transform=transform),
#                          batch_size=1,
#                          shuffle=False,
#                          drop_last=False,
#                          num_workers=2,
#                          prefetch_factor=2,
#                          pin_memory=True)

# for idx, (images, _) in enumerate(data_loader):
#     # new weights

