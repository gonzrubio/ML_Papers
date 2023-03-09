#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLIPstyler: Image Style Transfer with a Single Text Condition

paper: https://arxiv.org/abs/2112.00374

Created on Wed Feb 22 21:24:31 2023

@author: gonzalo
"""

# with torch.no_grad():
#     img_transformed = preprocess(img)
#     content_image = preprocess("path/to/content/image.jpg")
#     features = model(content_image)

#     features4_2 = model.features[:23](content_image)
#     features5_2 = model.features[:30](content_image)
#     content_loss = F.mse_loss(features4_2, features4_2) + F.mse_loss(features5_2, features5_2)


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


def stylize(img_c, txt):
    # get device
    # model = StyleNet()
    # for n_iter (look at paper)
    # forward
    # losses
    # update weights
    # display loss

    return img_cs


def load_vgg_clip():
    weights = VGG19_BN_Weights.DEFAULT
    preprocess_vgg = weights.transforms()
    vgg = vgg19_bn(weights=weights)
    vgg.eval()
    # load clip (from transformers or opeinai)
    return vgg, preprocess_vgg, clip, preprocess_clip


def get_dataloader(src):
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    # load images with bs=1 in torchvision Image class
    return dataloader


def get_conditions(src):
    return conditions


def main(cfg):
    """Text-guided style transfer pipeline."""
    # get text, images and pre-trained models
    conditions = get_conditions(cfg['text'])
    dataloader = get_dataloader(cfg['content'])
    vgg, preprocess_vgg, clip, preprocess_clip = load_vgg_clip()

    for name, module in model.named_modules():
        print(name, type(module))

    # for each image in loader tqdm
    # for each promtp tqdm (look up nice format nested tqdm)
    # img_cs = stylize(img_c, txt)
    # save img_c_txt.png (full res output/)
    # append to collage list

    # plot all images as table like fig 1 (highest res possible output/)


if __name__ == '__main__':
    # TODO look at style transfer favorite script
    # TODO look at paper and run on a single image-text condition pair
    # TODO see how authors use pre-trained models
    # TODO make content_images/ (x7) and make text_conditions/ (x7)
    # TODO read args and set defaults
    cfg= {
        'content': '',  # content image(s)
        'text' '',    # text condition(s)
        }

    main(cfg)
