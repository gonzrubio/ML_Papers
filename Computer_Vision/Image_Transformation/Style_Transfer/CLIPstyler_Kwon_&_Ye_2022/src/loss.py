#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 18:05:17 2023

@author: gonzalo
"""

import torch.nn.functional as F


def vgg_feature_maps(image, model):
    layers = {'21': 'conv4_2', '31': 'conv5_2'}
    features = {}
    x = image
    # x = x.unsqueeze(0)
    for name, layer in model._modules.items():
        x = layer(x)   
        if name in layers:
            features[layers[name]] = x
    
    return features


def content_loss(I_cs, vgg, transforms, vgg_features):
    # maintain content information of input image
    I_cs_vgg_features = vgg_feature_maps(transforms(I_cs), vgg)

    loss_content = F.mse_loss(vgg_features['conv4_2'],
                              I_cs_vgg_features['conv4_2'])
    loss_content += F.mse_loss(vgg_features['conv5_2'],
                              I_cs_vgg_features['conv5_2'])

    return loss_content