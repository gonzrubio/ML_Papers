#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLIPstyler: Image Style Transfer with a Single Text Condition

paper: https://arxiv.org/abs/2112.00374

Created on Wed Feb 22 21:24:31 2023

@author: gonzalo
"""

import os
import time

from PIL import Image, ImageOps
from tqdm import tqdm

import torch
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg19, VGG19_Weights
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import adjust_contrast
from torchvision.utils import save_image
torch.backends.cudnn.benchmark = True

import clip as openaiclip
import matplotlib.pyplot as plt

from StyleNet import StyleNet
from templates import compose_text_with_templates
from loss import vgg_feature_maps, clip_loss, content_loss, patch_loss, total_variation_loss


def stylize(img_c, txt, models, transforms, device):
    # start_time = time.time()

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
    n_patch = 64
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
        delta_T_patch = delta_T.repeat(n_patch, 1)
        # content image CLIP-space feature vector
        I_c_features = clip.encode_image(I_clip)
        I_c_features /= I_c_features.norm(dim=-1, keepdim=True)
        # content image VGG-space feature vector
        vgg_features = vgg_feature_maps(I_vgg, vgg)

    for i in range(200):
        # style transfer output (stylized content image)
        I_cs = stylenet(I_c)

        # losses
        loss_content = content_loss(I_cs, vgg, transforms['vgg'], vgg_features)
        loss_content *= lambdas['cont']

        loss_patch = patch_loss(I_cs, clip, transforms, I_c_features, delta_T_patch, n_patch)
        loss_patch *= lambdas['patch']

        loss_dir = clip_loss(I_cs, clip, transforms['clip'], I_c_features, delta_T)
        loss_dir *= lambdas['dir']

        loss_tv = lambdas['tv'] * total_variation_loss(I_cs)

        loss_total = loss_content + loss_patch + loss_dir + loss_tv

        # Print the losses
        # print(f"Iteration {i}: total loss = {total_loss.item()}, "
        #       f"dir loss = {loss_dir.item()}, patch loss = {loss_patch.item()}, "
        #       f"content loss = {loss_content.item()}, TV loss = {loss_tv.item()}")

        # gradients and update
        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        optimizer.step()
        scheduler.step()

    # elapsed_time = time.time() - start_time
    # print(f'\n elapsed time: {elapsed_time:.2f}s')

    return I_cs


def get_models_transforms():

    weights = VGG19_Weights.DEFAULT
    preprocess_vgg = weights.transforms()
    vgg = vgg19(weights=weights).features
    vgg.eval()

    clip, preprocess_clip = openaiclip.load('ViT-B/32', jit=False)
    clip.eval()

    for vgg_param, clip_param in zip(vgg.parameters(), clip.parameters()):
        vgg_param.requires_grad_(False)
        clip_param.requires_grad_(False)

    models = {'vgg': vgg, 'clip': clip}

    transforms = {
        'stylenet': T.Compose([
            T.Resize((1024, 1024)),
            T.ToTensor(),
            lambda x: torch.unsqueeze(x, 0)
            ]),
        'clip': T.Compose([
            lambda x: F.interpolate(x, size=224, mode='bicubic'),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711])
            ]),
        'patch': T.Compose([
            T.RandomCrop(size=256),
            T.RandomPerspective(fill=0, p=1, distortion_scale=0.5)
            ]),
        'vgg': T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
            ])
        # 'vgg': preprocess_vgg
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

    output_dir = os.path.join('..', 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(cfg['content'])):
        img_c = Image.open(os.path.join(cfg['content'], img_name))
        img_c = ImageOps.exif_transpose(img_c)

        for txt in tqdm(conditions, leave=False):

            img_cs = stylize(img_c, txt, models, transforms, device)

            img_cs = T.Resize((img_c.height, img_c.width), interpolation=InterpolationMode.BICUBIC)(img_cs)
            save_image(adjust_contrast(img_cs, 1.5),
                       os.path.join(output_dir, f'{img_name[:-4]}_{txt.replace(" ", "_")}.png'),
                       nrow=1,
                       normalize=True)


if __name__ == '__main__':

    cfg= {
        'content': os.path.join(os.getcwd(), '..', 'data', 'content_images'),
        'text': os.path.join(os.getcwd(), '..', 'data', 'text_conditions.txt')
        }

    main(cfg)
