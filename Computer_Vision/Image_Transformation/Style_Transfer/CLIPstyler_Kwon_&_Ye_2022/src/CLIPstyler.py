#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLIPstyler: Image Style Transfer with a Single Text Condition

paper: https://arxiv.org/abs/2112.00374

Created on Wed Feb 22 21:24:31 2023

@author: gonzalo
"""

import os

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg19, VGG19_Weights
from torchvision.transforms.functional import adjust_contrast

import clip as openaiclip
import matplotlib.pyplot as plt

from StyleNet import StyleNet
from templates import compose_text_with_templates


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


def stylize(img_c, txt, models, transforms, device):

    plt.imshow(adjust_contrast(img_c, 1.5))
    plt.show()

    clip = models['clip'].to(device)
    vgg = models['vgg'].to(device)
    stylenet = StyleNet().to(device)

    # optimization parameters
    optimizer = optim.Adam(stylenet.parameters(), lr=5e-4)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    lambdas = {'tv': 2e-3, 'patch': 9000, 'dir': 500, 'cont': 150}

    # prepare the inputs for StyleNet, VGG and CLIP
    token_t_sty = compose_text_with_templates(txt)
    token_t_sty = openaiclip.tokenize(token_t_sty).to(device)

    token_src = compose_text_with_templates('a Photo')
    token_src = openaiclip.tokenize(token_src).to(device)

    I_c = transforms['stylenet'](img_c).to(device)
    I_clip = transforms['clip'](I_c).to(device)
    I_vgg = transforms['vgg'](I_c).to(device)

    # get the CLIP (unit norm) and VGG embeddings
    with torch.no_grad():
        # text condition CLIP-space feature vector
        t_sty_features = clip.encode_text(token_t_sty).mean(dim=0, keepdim=True)
        t_sty_features /= t_sty_features.norm(dim=-1, keepdim=True)
        # source text ('a Photo') CLIP-space feature vector
        t_src_features = clip.encode_text(token_src).mean(dim=0, keepdim=True)
        t_src_features /= t_src_features.norm(dim=-1, keepdim=True)
        # direction of semantic text features
        delta_T = t_sty_features - t_src_features
        delta_T /= delta_T.norm(dim=-1, keepdim=True)
        # content image CLIP-space feature vector
        I_c_features = clip.encode_image(I_clip)
        I_c_features /= I_c_features.norm(dim=-1, keepdim=True)
        # content image VGG-space feature vector
        vgg_features = vgg_feature_maps(I_vgg, vgg)


    for i in range(200):

        # TODO reproduce their result and add losses one by one in order

        # directional CLIP loss
        # I_cs = stylenet(I_c)  # stylized content image
        # I_cs.requires_grad_(True)
        # plt.imshow(adjust_contrast(I_cs.clone(), 1.5).squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
        # plt.show()

        # I_cs_features = clip.encode_image(transforms['clip'](I_cs))
        # I_cs_features /= I_cs_features.clone().norm(dim=-1, keepdim=True)
        # delta_I = I_cs_features - I_c_features
        # delta_I /= delta_I.clone().norm(dim=-1, keepdim=True)
        # loss_dir = lambdas['dir'] * (1 - torch.cosine_similarity(delta_I, delta_T))

        # PatchCLIP loss
        # randomly crop n=64 patches and apply random geometrical (perspective sclae=0.5) augmentations and calculate clip loss
        # patch size 128
        # threshold rejection tau=0.7 (nullify loss for patches above some threshold)
        # loss_patch = lambda_p * ()

        # loss_content = lamda_c * MSE(img_c_vgg, img_cs_vgg)

        # loss_tv = lamda_tv * ()

        # loss_total = loss_dir + loss_patch + loss_content + loss_tv

        # Print the losses
        # print(f"Iteration {i}: total loss = {total_loss.item()}, "
        #       f"dir loss = {loss_dir.item()}, patch loss = {loss_patch.item()}, "
        #       f"content loss = {loss_content.item()}, TV loss = {loss_tv.item()}")
        # print(f'Iteration {i}, loss_dir: {loss_dir.item():.4f}')
        optimizer.zero_grad(set_to_none=True)
        loss_dir.backward()
        optimizer.step()
        scheduler.step()

    return img_cs


def get_models_transforms():

    weights = VGG19_Weights.DEFAULT
    preprocess_vgg = weights.transforms()
    vgg = vgg19(weights=weights).features
    vgg.eval()

    clip, preprocess_clip = openaiclip.load('ViT-B/32', jit=False)
    clip.eval()

    models = {'vgg': vgg, 'clip': clip}

    transforms = {
        'stylenet': T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            lambda x: torch.unsqueeze(x, 0)
            ]),
        'clip': T.Compose([
            lambda x: F.interpolate(x, size=224, mode='bicubic'),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711])
            ]),
        'vgg': preprocess_vgg,
        }

    return models, transforms


def get_conditions(src):
    with open(src, 'r') as file:
        conditions = [line.strip() for line in file.readlines()]
    return conditions


def main(cfg):
    """Text-guided style transfer pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conditions = get_conditions(cfg['text'])
    models, transforms = get_models_transforms()

    for img_c in tqdm(os.listdir(cfg['content'])):
        img_c = Image.open(os.path.join(cfg['content'], img_c))
        for txt in tqdm(conditions, leave=False):
            img_cs = stylize(img_c, txt, models, transforms, device)
            # save img_c_txt.png
            # append to collage list
    # plot all images as table (or two depending on size, fix num images per grid) like fig 1 (highest res possible output/)


if __name__ == '__main__':
    # TODO run end-to-end on a single image-text pair
    # TODO Look at official code
    # TODO your text_conditions/ (x7) and plots (add mexican gods to text)
    # TODO read args and set defaults
    cfg= {
        'content': os.path.join(os.getcwd(), '..', 'data', 'content_images'),
        'text': os.path.join(os.getcwd(), '..', 'data', 'text_conditions.txt')
        }

    main(cfg)
