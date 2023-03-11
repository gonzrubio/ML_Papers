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
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg19, VGG19_Weights

import clip as openaiclip
import matplotlib.pyplot as plt

from StyleNet import StyleNet


def feature_maps(image, model):

    layers = {'0': 'conv1_1',
              '5': 'conv2_1',
              '10': 'conv3_1',
              '19': 'conv4_1',
              '21': 'conv4_2',
              '28': 'conv5_1',
              '31': 'conv5_2'}

    features = {}
    x = image
    # x = x.unsqueeze(0)
    for name, layer in model._modules.items():
        x = layer(x)   
        if name in layers:
            features[layers[name]] = x
    
    return features


# with torch.no_grad():
#     img_transformed = preprocess(img)
#     content_image = preprocess("path/to/content/image.jpg")
#     features = model(content_image)

#     features4_2 = model.features[:23](content_image)
#     features5_2 = model.features[:30](content_image)
#     content_loss = F.mse_loss(features4_2, features4_2) + F.mse_loss(features5_2, features5_2)


def stylize(img_c, txt, vgg, preprocess_vgg, clip, preprocess_clip, device):
    # modularize as much a spossible
    lambda_d = 5e2
    lambda_p = 9e3
    lamda_c = 150
    lamda_tv = 2e-3

    stylenet = StyleNet()
    optimizer = optim.Adam(stylenet.parameters(), lr=5e-4)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

    for i in range(200):
        # modularize below in function to get embeddings (vgg, clip txt img)??

        # Prepare the inputs
        # TODO vgg inputs?
        # TODO image and (avg) tokenized text conditions embeddings
        text_tokens = torch.cat([openaiclip.tokenize(i) for i in txt]).to(device)

        # TODO Calculate clip and vgg features??
        with torch.no_grad():
            image_features = clip.encode_image(image_input)
            text_features = clip.encode_text(text_tokens).mean(dim=0)

        # Forward pass and losses
        img_cs = stylenet(img_c)
        # loss_dir = lambda_d * () # for semantic information
        # loss_patch = lambda_p * () # spatially invariant texture
        # randomly crop n=64 patches and apply random geometrical (perspective sclae=0.5) augmentations and calculate clip loss
        # patch size 128
        # threshold rejection tau=0.7 (nullify loss for patches above some threshold)
        # loss_content = lamda_c * MSE(img_c_vgg, img_cs_vgg)
        # loss_tv = lamda_tv * ()
        # loss_total = loss_dir + loss_patch + loss_content + loss_tv
        # Print the losses
        print(f"Iteration {i}: total loss = {total_loss.item()}, "
              f"dir loss = {loss_dir.item()}, patch loss = {loss_patch.item()}, "
              f"content loss = {loss_content.item()}, TV loss = {loss_tv.item()}")

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # scheduler.step()

    return img_cs


def load_vgg_clip():

    weights = VGG19_Weights.DEFAULT
    preprocess_vgg = weights.transforms()
    vgg = vgg19(weights=weights)
    vgg.eval()

    clip, preprocess_clip = openaiclip.load('ViT-B/32', jit=False)
    clip.eval()

    return vgg, preprocess_vgg, clip, preprocess_clip


def get_dataloader(src):

    class ImageFolder(Dataset):
        def __init__(self, data_path):
            self.data_path = data_path
            self.transforms = transforms.Compose([transforms.Resize((512, 512)),
                                                  transforms.ToTensor()])
            self.images = []
            for sample_name in os.listdir(self.data_path):
                sample_path = os.path.join(self.data_path, sample_name)
                self.images.append(sample_path)

        def __getitem__(self, idx):
            return self.transforms(Image.open(self.images[idx]))

        def __len__(self):
            return len(self.images)

    return DataLoader(ImageFolder(src))


def get_conditions(src):
    with open(src, 'r') as file:
        conditions = [line.strip().split(',') for line in file.readlines()]
    return conditions


def main(cfg):
    """Text-guided style transfer pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    conditions = get_conditions(cfg['text'])
    dataloader = get_dataloader(cfg['content'])
    vgg, preprocess_vgg, clip, preprocess_clip = load_vgg_clip()

    for img_c in tqdm(dataloader):
        for txt in tqdm(conditions, leave=False):
            img_cs = stylize(
                img_c, txt, vgg, preprocess_vgg, clip, preprocess_clip, device
                )
            # contrast enhancements
            # save img_c_txt.png
            # append to collage list

    # plot all images as table (or two depending on size, fix num images per grid) like fig 1 (highest res possible output/)


if __name__ == '__main__':
    # TODO run end-to-end on a single image-text pair
    # TODO more text_conditions/ (x7) and plots
    # TODO Look at official code
    # TODO read args and set defaults
    cfg= {
        'content': os.path.join(os.getcwd(), '..', 'data', 'content_images'),
        'text': os.path.join(os.getcwd(), '..', 'data', 'text_conditions.txt')
        }

    main(cfg)
