"""

Paper: 

Created on Wed Oct 20 13:38:44 2021

@author: gonzo
"""


import torch
import torch.nn as nn
import torchvision
import torch.optim as optim


from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import CosineAnnealingLR


device = "cuda:0" if torch.cuda.is_available() else "cpu"


##############################################################################
#                                                                            #
#                                    Data                                    #
#                                                                            #
##############################################################################

image_size = (398, 398)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
composed = transforms.Compose([transforms.Resize(image_size),
                               transforms.ToTensor(),
                               # transforms.Normalize(mean, std)
                               ])
standardize = transforms.Normalize(mean, std)

content_image = Image.open("Gonzalo.jpeg")
content_image = composed(content_image).unsqueeze(0)
content_image = content_image.to(device)

style_image = Image.open("pickle_rick.jpeg")
style_image = composed(style_image).unsqueeze(0)
style_image = style_image.to(device)

generated_image = content_image.clone().requires_grad_(True).to(device)


##############################################################################
#                                                                            #
#                                   Model                                    #
#                                                                            #
##############################################################################

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        self.conv_x_1_layers = ['0', '5', '10', '19', '28']

        # remove fc layers and freeze conv layers
        self.model = torchvision.models.vgg19(pretrained=True).features[:29]
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):

        feature_maps = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.conv_x_1_layers:
                feature_maps.append(x)

        return feature_maps


model = VGG19().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params:,}")

##############################################################################
#                                                                            #
#                                Hyperparameters                             #
#                                                                            #
##############################################################################

criterion = nn.CrossEntropyLoss()

num_iter = 10000
lr = 1e-3
alpha = 0.001
beta = 0.9

optimizer = optim.Adam([generated_image], lr=lr)
scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0)


##############################################################################
#                                                                            #
#                                  Training                                  #
#                                                                            #
##############################################################################

cont_images = model(content_image)
style_images = model(style_image)

for step in range(num_iter):

    gen_images = model(generated_image)

    content_loss = style_loss = 0
    for cont_f, style_f, gen_f in zip(cont_images, style_images, gen_images):

        content_loss = torch.mean((cont_f - gen_f)**2)
        style_loss = torch.mean(
            (style_f @ style_f.permute(0, 1, 3, 2) -
             gen_f @ gen_f.permute(0, 1, 3, 2)) ** 2)

    loss = alpha * content_loss + beta * style_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(f"step {step}, loss {loss}")

save_image(generated_image, "Gonzalo_pickle.jpeg")
