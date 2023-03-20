#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 18:05:17 2023

@author: gonzalo
"""

import torch
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


def patch_loss(I_cs, clip, transforms, I_c_features, delta_T_patch, n_patch):
    # extract patches
    I_cs_patch = [transforms['patch'](I_cs) for p in range(n_patch)]
    I_cs_patch = torch.cat(I_cs_patch, dim=0)

    # get their CLIP-space feature vector
    I_cs_patch_feat = clip.encode_image(transforms['clip'](I_cs_patch))
    I_cs_patch_feat = I_cs_patch_feat / I_cs_patch_feat.norm(dim=-1, keepdim=True)

    # compute the direction of semantic stylized patch features
    delta_I_patch = I_cs_patch_feat - I_c_features
    delta_I_patch = delta_I_patch / delta_I_patch.norm(dim=-1, keepdim=True)
    loss_temp = 1 - torch.cosine_similarity(delta_I_patch, delta_T_patch, dim=1)
    loss_temp[loss_temp < 0.7] = 0

    return loss_temp.mean()
