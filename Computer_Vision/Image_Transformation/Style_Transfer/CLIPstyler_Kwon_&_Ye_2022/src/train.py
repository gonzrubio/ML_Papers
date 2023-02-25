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



# TODO vgg from torchvision/ (why use those specific layers?)
# TODO ask chatgpt if I can use another SOTA convnet or transformer

from torchvision.models import ViT_H_14_Weights, vit_h_14
import torch.nn.functional as F
with torch.no_grad():
    weights = ViT_H_14_Weights.DEFAULT
    preprocess = weights.transforms()
    model = vit_h_14(weights=weights)
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

